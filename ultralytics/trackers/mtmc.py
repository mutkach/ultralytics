# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Multi-Target Multi-Camera (MTMC) tracking bridge for coordinating multiple BOTSORT trackers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .utils.matching import embedding_distance, linear_assignment

if TYPE_CHECKING:
    from .bot_sort import BOTrack, BOTSORT


class MTMCBridge:
    """
    Coordinates multiple BOTSORT trackers for cross-camera tracking.

    This class manages multiple trackers (one per camera) and provides functionality for:
    - Automatic global ID unification when ReID embeddings match across cameras
    - Face verification callback interface for external identity verification
    - Alarm detection for unverified tracks exceeding a timeout threshold

    Attributes:
        trackers (dict): Mapping of camera_id to tracker instance.
        reid_threshold (float): Cosine distance threshold for cross-camera ReID matching (lower = stricter).
        min_tracklet_len (int): Minimum track length before considering for cross-camera matching.
        alarm_timeout (int): Number of frames before an unverified track triggers an alarm.

    Examples:
        Basic usage with two cameras:
        >>> from ultralytics.trackers import BOTSORT, MTMCBridge
        >>> tracker1 = BOTSORT(args, frame_rate=30)
        >>> tracker2 = BOTSORT(args, frame_rate=30)
        >>> bridge = MTMCBridge(reid_threshold=0.3, alarm_timeout_frames=600)
        >>> bridge.register("cam1", tracker1)
        >>> bridge.register("cam2", tracker2)
        >>> # After updating trackers with detections...
        >>> bridge.update()  # Matches tracks across cameras
        >>> alarms = bridge.get_alarms()  # Get unverified tracks
    """

    def __init__(
        self,
        reid_threshold: float = 0.3,
        same_camera_threshold: float = 0.2,
        min_tracklet_len: int = 5,
        alarm_timeout_frames: int = 600,
    ):
        """
        Initialize MTMCBridge for cross-camera track coordination.

        Args:
            reid_threshold (float): Cosine distance threshold for cross-camera ReID matching.
                Lower values are stricter (0.3 = 70% similarity required). Cross-camera matching
                typically needs looser thresholds due to lighting/angle differences.
            same_camera_threshold (float): Cosine distance threshold for same-camera matching.
                Reserved for future use. Stricter than cross-camera (0.2 = 80% similarity).
            min_tracklet_len (int): Minimum number of frames a track must exist before being
                considered for cross-camera matching. Allows smooth_feat to stabilize.
            alarm_timeout_frames (int): Number of frames before unverified track triggers alarm.
                At 30fps, 600 frames = 20 seconds.
        """
        self.trackers: dict[str, BOTSORT] = {}
        self.reid_threshold = reid_threshold
        self.same_camera_threshold = same_camera_threshold
        self.min_tracklet_len = min_tracklet_len
        self.alarm_timeout = alarm_timeout_frames
        self._global_id_counter = 0

    def register(self, camera_id: str, tracker: BOTSORT) -> None:
        """
        Register a tracker for a camera.

        Args:
            camera_id (str): Unique identifier for the camera.
            tracker (BOTSORT): BOTSORT tracker instance for this camera.
        """
        self.trackers[camera_id] = tracker

    def unregister(self, camera_id: str) -> bool:
        """
        Unregister a camera's tracker.

        Args:
            camera_id (str): Camera identifier to remove.

        Returns:
            bool: True if camera was registered and removed, False otherwise.
        """
        if camera_id in self.trackers:
            del self.trackers[camera_id]
            return True
        return False

    def update(self) -> None:
        """
        Update cross-camera matching and verification status.

        This method should be called after all individual trackers have been updated
        with their respective detections. It performs:
        1. Cross-camera track matching using ReID embeddings
        2. Global ID unification for matched tracks
        3. Alarm status updates for unverified tracks exceeding timeout
        """
        self._match_across_cameras()
        self._check_alarms()

    def _get_all_tracks(self) -> list[BOTrack]:
        """
        Get all active tracks from all registered cameras.

        Returns:
            list[BOTrack]: List of all tracked objects with camera_id tagged.
        """
        tracks = []
        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                track.camera_id = cam_id  # Tag track with its camera source
                tracks.append(track)
        return tracks

    def _match_across_cameras(self) -> None:
        """Match tracks across cameras using ReID embeddings and unify global IDs."""
        cam_ids = list(self.trackers.keys())

        # Compare each pair of cameras
        for i, cam1 in enumerate(cam_ids):
            for cam2 in cam_ids[i + 1 :]:
                # Get tracks with valid embeddings and sufficient length for stable matching
                tracks1 = [
                    t
                    for t in self.trackers[cam1].tracked_stracks
                    if t.smooth_feat is not None and t.tracklet_len >= self.min_tracklet_len
                ]
                tracks2 = [
                    t
                    for t in self.trackers[cam2].tracked_stracks
                    if t.smooth_feat is not None and t.tracklet_len >= self.min_tracklet_len
                ]

                if not tracks1 or not tracks2:
                    continue

                # Compute embedding distance matrix
                dists = embedding_distance(tracks1, tracks2)

                # One-to-one matching using Hungarian algorithm
                matches, _, _ = linear_assignment(dists, thresh=self.reid_threshold)
                for idx1, idx2 in matches:
                    self._unify_ids(tracks1[idx1], tracks2[idx2])

    def _unify_ids(self, track1: BOTrack, track2: BOTrack) -> None:
        """
        Assign the same global_id to two matching tracks.

        If one track already has a global_id, propagate it to the other.
        If neither has one, create a new global_id for both.

        Args:
            track1 (BOTrack): First track to unify.
            track2 (BOTrack): Second track to unify.
        """
        if track1.global_id is not None:
            track2.global_id = track1.global_id
        elif track2.global_id is not None:
            track1.global_id = track2.global_id
        else:
            self._global_id_counter += 1
            track1.global_id = self._global_id_counter
            track2.global_id = self._global_id_counter

    def _check_alarms(self) -> None:
        """Mark unverified tracks as 'alarm' if they exceed the timeout threshold."""
        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                if track.verification_status == "waiting" and track.tracklet_len > self.alarm_timeout:
                    track.verification_status = "alarm"

    def verify_track(self, camera_id: str, track_id: int, face_id: str) -> bool:
        """
        Mark a track as verified with a face ID.

        Args:
            camera_id (str): Camera where the track is located.
            track_id (int): Local track ID within that camera's tracker.
            face_id (str): External face identity string from face recognition.

        Returns:
            bool: True if track was found and verified, False otherwise.
        """
        tracker = self.trackers.get(camera_id)
        if tracker:
            for track in tracker.tracked_stracks:
                if track.track_id == track_id:
                    track.verification_status = "confirmed"
                    track.face_id = face_id
                    return True
        return False

    def verify_by_global_id(self, global_id: int, face_id: str) -> int:
        """
        Mark all tracks with a given global_id as verified.

        This is useful when face verification succeeds on one camera and you want
        to propagate the verification to all matched tracks across cameras.

        Args:
            global_id (int): Global ID to search for across all cameras.
            face_id (str): External face identity string from face recognition.

        Returns:
            int: Number of tracks that were verified.
        """
        count = 0
        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                if track.global_id == global_id:
                    track.verification_status = "confirmed"
                    track.face_id = face_id
                    count += 1
        return count

    def get_tracks(self, camera_id: str | None = None) -> list[BOTrack]:
        """
        Get tracks, optionally filtered by camera.

        Args:
            camera_id (str | None): If provided, only return tracks from this camera.
                If None, return tracks from all cameras.

        Returns:
            list[BOTrack]: List of tracks (with camera_id attribute set).
        """
        if camera_id is not None:
            tracker = self.trackers.get(camera_id)
            if tracker:
                tracks = list(tracker.tracked_stracks)
                for track in tracks:
                    track.camera_id = camera_id
                return tracks
            return []
        return self._get_all_tracks()

    def get_alarms(self) -> list[BOTrack]:
        """
        Get all tracks in alarm state.

        Returns:
            list[BOTrack]: List of tracks with verification_status == "alarm".
        """
        return [t for t in self._get_all_tracks() if t.verification_status == "alarm"]

    def get_waiting(self) -> list[BOTrack]:
        """
        Get all tracks in waiting state.

        Returns:
            list[BOTrack]: List of tracks with verification_status == "waiting".
        """
        return [t for t in self._get_all_tracks() if t.verification_status == "waiting"]

    def get_confirmed(self) -> list[BOTrack]:
        """
        Get all verified/confirmed tracks.

        Returns:
            list[BOTrack]: List of tracks with verification_status == "confirmed".
        """
        return [t for t in self._get_all_tracks() if t.verification_status == "confirmed"]

    def get_track_by_id(self, camera_id: str, track_id: int) -> BOTrack | None:
        """
        Get a specific track by camera and track ID.

        Args:
            camera_id (str): Camera identifier.
            track_id (int): Local track ID.

        Returns:
            BOTrack | None: The track if found, None otherwise.
        """
        tracker = self.trackers.get(camera_id)
        if tracker:
            for track in tracker.tracked_stracks:
                if track.track_id == track_id:
                    track.camera_id = camera_id
                    return track
        return None

    def get_tracks_by_global_id(self, global_id: int) -> list[BOTrack]:
        """
        Get all tracks sharing a global ID across cameras.

        Args:
            global_id (int): Global ID to search for.

        Returns:
            list[BOTrack]: List of tracks with matching global_id.
        """
        return [t for t in self._get_all_tracks() if t.global_id == global_id]

    def reset(self) -> None:
        """Reset all trackers and clear the global ID counter."""
        for tracker in self.trackers.values():
            tracker.reset()
        self._global_id_counter = 0

    def __len__(self) -> int:
        """Return total number of active tracks across all cameras."""
        return sum(len(tracker.tracked_stracks) for tracker in self.trackers.values())

    def __repr__(self) -> str:
        """Return string representation of MTMCBridge state."""
        return (
            f"MTMCBridge(cameras={list(self.trackers.keys())}, "
            f"tracks={len(self)}, "
            f"reid_threshold={self.reid_threshold}, "
            f"same_camera_threshold={self.same_camera_threshold}, "
            f"min_tracklet_len={self.min_tracklet_len}, "
            f"alarm_timeout={self.alarm_timeout})"
        )
