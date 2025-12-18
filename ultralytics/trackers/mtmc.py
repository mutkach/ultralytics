# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from .identity_store import IdentityStore, StoredIdentity, TrackGallery
from .utils.matching import linear_assignment

if TYPE_CHECKING:
    from .bot_sort import BOTrack, BOTSORT


def check_crop_quality(crop: np.ndarray,
                       min_brightness: float = 40.0,
                       min_saturation: float = 20.0,
                       center_ratio: float = 0.5) -> bool:
    """Check if crop has sufficient lighting quality for ReID.

    Focuses on center region to avoid being fooled by bright backgrounds
    when the person is poorly lit.

    Args:
        crop: RGB image array
        min_brightness: Minimum mean brightness of center region (0-255)
        min_saturation: Minimum mean saturation of center region (0-255)
        center_ratio: Ratio of crop to use for center region (0.5 = middle 50%)

    Returns:
        True if crop has sufficient quality for ReID
    """
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return False

    # Extract center region (where the body should be)
    margin_x = int(w * (1 - center_ratio) / 2)
    margin_y = int(h * (1 - center_ratio) / 2)
    center = crop[margin_y:h-margin_y, margin_x:w-margin_x]

    if center.size == 0:
        return False

    # Check brightness of center region
    gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) < min_brightness:
        return False

    # Check color saturation of center (dark areas are desaturated)
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    if np.mean(hsv[:, :, 1]) < min_saturation:
        return False

    return True


class MTMCBridge:
    def __init__(
        self,
        reid_threshold: float = 0.3,
        min_tracklet_len: int = 5,
        alarm_timeout_frames: int = 600,
        identity_match_threshold: float = 0.25,
        enable_identity_matching: bool = True,
        max_features_per_identity: int = 10,
        identity_ttl_frames: int = 108000,
        track_gallery_threshold: float = 0.15,
        enable_track_gallery: bool = True,
        debug: bool = False,
        crop_store=None,
        min_brightness: float = 40.0,
        min_saturation: float = 20.0,
        quality_center_ratio: float = 0.5,
    ):
        self.trackers: dict[str, BOTSORT] = {}
        self.reid_threshold = reid_threshold
        self.min_tracklet_len = min_tracklet_len
        self.debug = debug
        self.alarm_timeout = alarm_timeout_frames
        self._global_id_counter = 0
        self.crop_store = crop_store
        self.min_brightness = min_brightness
        self.min_saturation = min_saturation
        self.quality_center_ratio = quality_center_ratio

        self.enable_identity_matching = enable_identity_matching
        self.identity_store = IdentityStore(
            max_features_per_identity=max_features_per_identity,
            match_threshold=identity_match_threshold,
            feature_ttl_frames=identity_ttl_frames,
        )

        self.enable_track_gallery = enable_track_gallery
        self.track_gallery = TrackGallery(
            max_features_per_track=max_features_per_identity,
            match_threshold=track_gallery_threshold,
            track_ttl_frames=identity_ttl_frames,
        )

    def register(self, camera_id: str, tracker: BOTSORT) -> None:
        self.trackers[camera_id] = tracker

    def unregister(self, camera_id: str) -> bool:
        if camera_id in self.trackers:
            del self.trackers[camera_id]
            return True
        return False

    def update(self, frame_num: int = 0, frames: Optional[dict] = None) -> None:
        self._match_across_cameras()
        self._match_against_track_gallery(frame_num)
        self._assign_global_ids()
        self._match_against_identities()
        self._update_track_gallery(frame_num, frames)
        self._check_alarms()

        if frame_num > 0 and frame_num % 1000 == 0:
            self.identity_store.cleanup(frame_num)
            self.track_gallery.cleanup(frame_num)

    def _get_all_tracks(self) -> list[BOTrack]:
        tracks = []
        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                track.camera_id = cam_id
                tracks.append(track)
        return tracks

    def _match_across_cameras(self) -> None:
        cam_ids = list(self.trackers.keys())

        for i, cam1 in enumerate(cam_ids):
            for cam2 in cam_ids[i + 1:]:
                tracks1 = [
                    t for t in self.trackers[cam1].tracked_stracks
                    if t.smooth_feat is not None and t.tracklet_len >= self.min_tracklet_len
                ]
                tracks2 = [
                    t for t in self.trackers[cam2].tracked_stracks
                    if t.smooth_feat is not None and t.tracklet_len >= self.min_tracklet_len
                ]

                if not tracks1 or not tracks2:
                    continue

                feats1 = np.asarray([t.smooth_feat for t in tracks1], dtype=np.float32)
                feats2 = np.asarray([t.smooth_feat for t in tracks2], dtype=np.float32)
                dists = cdist(feats1, feats2, metric='cosine')

                matches, _, _ = linear_assignment(dists, thresh=self.reid_threshold)
                for idx1, idx2 in matches:
                    t1, t2 = tracks1[idx1], tracks2[idx2]
                    dist = dists[idx1, idx2]
                    self._unify_tracks(t1, t2)
                    if self.debug:
                        print(
                            f"[MTMC] Cross-cam: {cam1}:T{t1.track_id} <-> {cam2}:T{t2.track_id} | "
                            f"dist={dist:.4f} (thresh={self.reid_threshold}) -> G:{t1.global_id}"
                        )

    def _match_against_track_gallery(self, frame_num: int) -> None:
        if not self.enable_track_gallery or len(self.track_gallery) == 0:
            return

        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                if track.global_id is not None:
                    continue
                if track.smooth_feat is None or track.tracklet_len < self.min_tracklet_len:
                    continue

                result = self.track_gallery.match(track.smooth_feat)
                if result:
                    global_id, distance, face_id = result
                    track.global_id = global_id
                    if face_id:
                        track.verification_status = "confirmed"
                        track.face_id = face_id
                        track._reid_confirmed = True
                    if self.debug:
                        print(
                            f"[MTMC] Gallery: T{track.track_id} -> G:{global_id} | "
                            f"dist={distance:.4f} (thresh={self.track_gallery.match_threshold}) "
                            f"face={face_id or 'none'}"
                        )

    def _assign_global_ids(self) -> None:
        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                if track.global_id is not None:
                    continue
                if track.tracklet_len < self.min_tracklet_len:
                    continue
                self._global_id_counter += 1
                track.global_id = self._global_id_counter
                if self.debug:
                    print(f"[MTMC] New ID: {cam_id}:T{track.track_id} -> G:{track.global_id} (no match found)")

    def _update_track_gallery(self, frame_num: int, frames: Optional[dict] = None) -> None:
        if not self.enable_track_gallery:
            return

        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                if track.global_id is None:
                    continue
                if track.smooth_feat is None:
                    continue

                # Extract crop for contrast check (if frames provided)
                crop = None
                if frames is not None:
                    frame = frames.get(cam_id)
                    if frame is not None:
                        x1, y1, x2, y2 = map(int, track.xyxy)
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]

                # Skip low-quality crops (dark, desaturated)
                if crop is not None and self.min_brightness > 0:
                    if not check_crop_quality(crop, self.min_brightness,
                                              self.min_saturation,
                                              self.quality_center_ratio):
                        continue

                added = self.track_gallery.register(
                    track.global_id,
                    track.smooth_feat,
                    frame_num,
                    track.face_id,
                )

                # Save crop when feature is added to gallery
                if added and self.crop_store is not None and crop is not None:
                    self.crop_store.save(
                        crop,
                        global_id=track.global_id,
                        frame_num=frame_num,
                        camera_id=cam_id,
                        track_id=track.track_id,
                        face_id=track.face_id,
                    )

    def _unify_tracks(self, track1: BOTrack, track2: BOTrack) -> None:
        old_gid1, old_gid2 = track1.global_id, track2.global_id

        if track1.global_id is not None and track2.global_id is not None:
            if track1.global_id != track2.global_id:
                keep_gid = min(track1.global_id, track2.global_id)
                discard_gid = max(track1.global_id, track2.global_id)
                track1.global_id = keep_gid
                track2.global_id = keep_gid
                for tracker in self.trackers.values():
                    for t in tracker.tracked_stracks:
                        if t.global_id == discard_gid:
                            t.global_id = keep_gid
                self.track_gallery.merge(discard_gid, keep_gid)
        elif track1.global_id is not None:
            track2.global_id = track1.global_id
        elif track2.global_id is not None:
            track1.global_id = track2.global_id
        else:
            self._global_id_counter += 1
            track1.global_id = self._global_id_counter
            track2.global_id = self._global_id_counter

        if track1.verification_status == "confirmed" and track2.verification_status != "confirmed":
            track2.verification_status = "confirmed"
            if track1.face_id:
                track2.face_id = track1.face_id
                track2._reid_confirmed = True
        elif track2.verification_status == "confirmed" and track1.verification_status != "confirmed":
            track1.verification_status = "confirmed"
            if track2.face_id:
                track1.face_id = track2.face_id
                track1._reid_confirmed = True

        if track1.face_id and not track2.face_id:
            track2.face_id = track1.face_id
        elif track2.face_id and not track1.face_id:
            track1.face_id = track2.face_id

    def _check_alarms(self) -> None:
        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                if track.verification_status == "waiting" and track.tracklet_len > self.alarm_timeout:
                    track.verification_status = "alarm"
                    if self.debug:
                        print(
                            f"[MTMC] Alarm: {cam_id}:T{track.track_id} G:{track.global_id} | "
                            f"frames={track.tracklet_len} (timeout={self.alarm_timeout})"
                        )

    def _match_against_identities(self) -> None:
        if not self.enable_identity_matching or len(self.identity_store) == 0:
            return

        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                if track.verification_status != "waiting":
                    continue
                if track.smooth_feat is None or track.tracklet_len < self.min_tracklet_len:
                    continue

                result = self.identity_store.match(track.smooth_feat)
                if result:
                    face_id, distance = result
                    track.verification_status = "confirmed"
                    track.face_id = face_id
                    track._reid_confirmed = True
                    if self.debug:
                        print(
                            f"[MTMC] Identity: T{track.track_id} -> {face_id} | "
                            f"dist={distance:.4f} (thresh={self.identity_store.match_threshold}) -> CONFIRMED"
                        )

    def verify_track(self, camera_id: str, track_id: int, face_id: str, frame_num: int = 0) -> bool:
        tracker = self.trackers.get(camera_id)
        if tracker:
            for track in tracker.tracked_stracks:
                if track.track_id == track_id:
                    if track.face_id and track.face_id != face_id:
                        print(f"[MTMC] Identity conflict: reID said '{track.face_id}', face says '{face_id}'. Face wins.")

                    track.verification_status = "confirmed"
                    track.face_id = face_id
                    track._reid_confirmed = False

                    if track.smooth_feat is not None:
                        self.identity_store.register(face_id, track.smooth_feat, frame_num, track.global_id)

                    if track.global_id is not None:
                        self.track_gallery.update_face_id(track.global_id, face_id)

                    return True
        return False

    def verify_by_global_id(self, global_id: int, face_id: str, frame_num: int = 0) -> int:
        count = 0
        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                if track.global_id == global_id:
                    track.verification_status = "confirmed"
                    track.face_id = face_id
                    track._reid_confirmed = False

                    if track.smooth_feat is not None:
                        self.identity_store.register(face_id, track.smooth_feat, frame_num, track.global_id)
                    count += 1

        self.track_gallery.update_face_id(global_id, face_id)
        return count

    def get_tracks(self, camera_id: str | None = None) -> list[BOTrack]:
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
        return [t for t in self._get_all_tracks() if t.verification_status == "alarm"]

    def get_waiting(self) -> list[BOTrack]:
        return [t for t in self._get_all_tracks() if t.verification_status == "waiting"]

    def get_confirmed(self) -> list[BOTrack]:
        return [t for t in self._get_all_tracks() if t.verification_status == "confirmed"]

    def get_track_by_id(self, camera_id: str, track_id: int) -> BOTrack | None:
        tracker = self.trackers.get(camera_id)
        if tracker:
            for track in tracker.tracked_stracks:
                if track.track_id == track_id:
                    track.camera_id = camera_id
                    return track
        return None

    def get_tracks_by_global_id(self, global_id: int) -> list[BOTrack]:
        return [t for t in self._get_all_tracks() if t.global_id == global_id]

    def get_known_identities(self) -> list[str]:
        return self.identity_store.get_all_identities()

    def get_identity_info(self, face_id: str) -> Optional[StoredIdentity]:
        return self.identity_store.get_identity(face_id)

    def reset(self) -> None:
        for tracker in self.trackers.values():
            tracker.reset()
        self._global_id_counter = 0
        self.identity_store.clear()
        self.track_gallery.clear()

    def __len__(self) -> int:
        return sum(len(tracker.tracked_stracks) for tracker in self.trackers.values())

    def __repr__(self) -> str:
        return (
            f"MTMCBridge(cameras={list(self.trackers.keys())}, "
            f"tracks={len(self)}, "
            f"identities={len(self.identity_store)}, "
            f"gallery={len(self.track_gallery)}, "
            f"reid_threshold={self.reid_threshold})"
        )
