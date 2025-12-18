# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np

from .identity_store import FaceGallery, StoredFaceIdentity

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
    """Multi-Target Multi-Camera tracking bridge using face vectors.

    This class coordinates multiple BOTSORT trackers for cross-camera tracking
    using face vectors (InsightFace) as the primary method for identity association.
    Global IDs are only assigned after successful face detection.

    Args:
        face_app: InsightFace FaceAnalysis instance for face detection/embedding
        face_high_threshold: Cosine distance threshold for confident match (default: 0.5)
        face_low_threshold: Cosine distance threshold for tentative match (default: 0.7)
        min_tracklet_len: Minimum frames before face detection attempt (default: 5)
        alarm_timeout_frames: Frames before alarm for unverified tracks (default: 600)
        face_detection_interval: Frames between face detection attempts (default: 10)
        max_vectors_per_identity: Maximum face vectors per identity (default: 10)
        vector_ttl_frames: Time-to-live for face vectors (default: 108000)
        min_face_det_score: Minimum face detection confidence (default: 0.5)
        min_crop_size: Minimum crop dimensions for face detection (default: (64, 64))
        face_verifier: Optional external FaceVerifier for database lookup
        debug: Enable debug logging (default: False)
        crop_store: Optional crop storage for debugging
    """

    def __init__(
        self,
        face_app: Any = None,
        face_high_threshold: float = 0.5,
        face_low_threshold: float = 0.7,
        min_tracklet_len: int = 5,
        alarm_timeout_frames: int = 600,
        face_detection_interval: int = 10,
        max_vectors_per_identity: int = 10,
        vector_ttl_frames: int = 108000,
        min_face_det_score: float = 0.5,
        min_crop_size: tuple = (64, 64),
        face_verifier: Any = None,
        debug: bool = False,
        crop_store: Any = None,
    ):
        self.trackers: dict[str, BOTSORT] = {}
        self.face_app = face_app
        self.face_verifier = face_verifier
        self.debug = debug
        self.crop_store = crop_store

        # Thresholds
        self.face_high_threshold = face_high_threshold
        self.face_low_threshold = face_low_threshold
        self.min_tracklet_len = min_tracklet_len
        self.alarm_timeout = alarm_timeout_frames
        self.face_detection_interval = face_detection_interval
        self.min_face_det_score = min_face_det_score
        self.min_crop_size = min_crop_size

        # Face gallery for cross-camera matching
        self.face_gallery = FaceGallery(
            max_vectors_per_identity=max_vectors_per_identity,
            high_threshold=face_high_threshold,
            low_threshold=face_low_threshold,
            vector_ttl_frames=vector_ttl_frames,
        )

        # Track last face detection attempt per track
        self._last_face_attempt: dict[int, int] = {}

    def register(self, camera_id: str, tracker: BOTSORT) -> None:
        """Register a camera tracker with this bridge."""
        self.trackers[camera_id] = tracker

    def unregister(self, camera_id: str) -> bool:
        """Unregister a camera tracker."""
        if camera_id in self.trackers:
            del self.trackers[camera_id]
            return True
        return False

    def update(self, frame_num: int = 0, frames: Optional[dict] = None) -> None:
        """Main update loop - face-vector based matching.

        Pipeline stages:
        1. Tag tracks with camera_id
        2. Attempt face detection on eligible tracks
        3. Match face vectors against gallery (cross-camera matching)
        4. External face database verification (if verifier provided)
        5. Update gallery with new vectors
        6. Cross-camera propagation (same global_id across cameras)
        7. Alarm check for tracks without faces

        Args:
            frame_num: Current frame number
            frames: Dict mapping camera_id to frame image (required for face detection)
        """
        # Stage 1: Tag all tracks with camera_id
        self._tag_camera_ids()

        # Stage 2: Face detection on eligible tracks
        if frames is not None and self.face_app is not None:
            self._detect_faces(frame_num, frames)

        # Stage 3: Match face vectors against gallery
        self._match_face_vectors(frame_num)

        # Stage 4: External face database verification (if verifier provided)
        if self.face_verifier is not None:
            self._verify_against_database(frame_num)

        # Stage 5: Update gallery with new vectors from tracked identities
        self._update_face_gallery(frame_num)

        # Stage 6: Cross-camera propagation
        self._propagate_across_cameras()

        # Stage 7: Alarm check
        self._check_alarms()

        # Periodic cleanup
        if frame_num > 0 and frame_num % 1000 == 0:
            self.face_gallery.cleanup(frame_num)
            # Clean up stale face attempt records
            self._cleanup_face_attempts()

    def _tag_camera_ids(self) -> None:
        """Set camera_id on all tracks."""
        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                track.camera_id = cam_id

    def _get_all_tracks(self) -> list[BOTrack]:
        """Get all tracks from all cameras."""
        tracks = []
        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                track.camera_id = cam_id
                tracks.append(track)
        return tracks

    def _detect_faces(self, frame_num: int, frames: dict) -> None:
        """Detect faces in person crops and extract embeddings.

        Only attempts on tracks that:
        - Have tracklet_len >= min_tracklet_len
        - Haven't had recent face detection attempt
        - Have sufficient crop size
        """
        for cam_id, frame in frames.items():
            tracker = self.trackers.get(cam_id)
            if tracker is None:
                continue

            for track in tracker.tracked_stracks:
                # Skip if not mature enough
                if track.tracklet_len < self.min_tracklet_len:
                    continue

                # Skip if recently attempted
                last_attempt = self._last_face_attempt.get(track.track_id, 0)
                if frame_num - last_attempt < self.face_detection_interval:
                    continue

                # Skip if already has global_id and face_vector (already matched)
                # But still try to detect faces for tracks without global_id
                if track.global_id is not None and track.face_vector is not None:
                    continue

                # Extract crop
                x1, y1, x2, y2 = map(int, track.xyxy)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Check minimum size
                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w < self.min_crop_size[0] or crop_h < self.min_crop_size[1]:
                    if self.debug:
                        print(f"[MTMC] Crop too small: {cam_id}:T{track.track_id} {crop_w}x{crop_h} < {self.min_crop_size}")
                    continue

                crop = frame[y1:y2, x1:x2]
                self._last_face_attempt[track.track_id] = frame_num

                # Run face detection
                try:
                    faces = self.face_app.get(crop)
                except Exception as e:
                    if self.debug:
                        print(f"[MTMC] Face detection error for {cam_id}:T{track.track_id}: {e}")
                    continue

                if len(faces) == 0:
                    if self.debug:
                        print(f"[MTMC] No face found: {cam_id}:T{track.track_id} (crop {crop_w}x{crop_h})")
                    continue

                # Take best face by detection score
                best_face = max(faces, key=lambda f: f.det_score)
                if best_face.det_score < self.min_face_det_score:
                    continue

                # Store face vector on track (copy to avoid reference issues)
                track.face_vector = np.array(best_face.embedding, dtype=np.float32)
                track.face_detection_frame = frame_num

                # Save crop for debugging
                if self.crop_store is not None:
                    self.crop_store.save(
                        crop,
                        global_id=track.global_id,
                        frame_num=frame_num,
                        camera_id=cam_id,
                        track_id=track.track_id,
                        face_id=track.face_id,
                    )

                if self.debug:
                    print(
                        f"[MTMC] Face detected: {cam_id}:T{track.track_id} | "
                        f"det_score={best_face.det_score:.2f}"
                    )

    def _match_face_vectors(self, frame_num: int) -> None:
        """Match tracks with face vectors against gallery.

        Two-threshold system:
        - distance < high_threshold: Confident match, assign global_id
        - high_threshold <= distance < low_threshold: Tentative, store vector anyway
        - distance >= low_threshold: No match, create new identity
        """
        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                # Skip tracks without face vectors
                if track.face_vector is None:
                    continue

                # Skip tracks that already have global_id
                if track.global_id is not None:
                    continue

                # Match against gallery
                result = self.face_gallery.match(track.face_vector)

                if result is not None:
                    global_id, distance, face_id = result

                    if distance < self.face_high_threshold:
                        # Confident match
                        track.global_id = global_id
                        if face_id:
                            track.face_id = face_id
                            track.verification_status = "confirmed"

                        if self.debug:
                            print(
                                f"[MTMC] Face match: {cam_id}:T{track.track_id} -> G:{global_id} | "
                                f"dist={distance:.4f} (high_thresh={self.face_high_threshold}) "
                                f"face={face_id or 'none'}"
                            )
                    else:
                        # Tentative match (low_threshold > distance >= high_threshold)
                        # TODO: explore quality metrics to decide whether to store vector
                        self.face_gallery.add_vector(global_id, track.face_vector, frame_num)

                        if self.debug:
                            print(
                                f"[MTMC] Face tentative: {cam_id}:T{track.track_id} | "
                                f"dist={distance:.4f} (low_thresh={self.face_low_threshold}) - vector stored"
                            )
                else:
                    # No match - create new identity
                    new_global_id = self.face_gallery.register_new(track.face_vector, frame_num)
                    track.global_id = new_global_id

                    if self.debug:
                        print(f"[MTMC] New face identity: {cam_id}:T{track.track_id} -> G:{new_global_id}")

    def _verify_against_database(self, frame_num: int) -> None:
        """Verify tracks against external face database.

        Uses the face_verifier to check if track's face matches known identities
        in the external database.
        """
        if self.face_verifier is None:
            return

        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                # Only verify waiting tracks with global_id
                if track.verification_status != "waiting":
                    continue
                if track.global_id is None:
                    continue
                if track.face_vector is None:
                    continue

                # Use face_vector directly with verify_embedding
                try:
                    face_id = self.face_verifier.verify_embedding(track.face_vector)
                except Exception as e:
                    if self.debug:
                        print(f"[MTMC] Database verification error for {cam_id}:T{track.track_id}: {e}")
                    continue

                if self.debug and face_id == "Not found":
                    print(f"[MTMC] Database no match: {cam_id}:T{track.track_id} G:{track.global_id}")

                if face_id and face_id != "Not found":
                    track.verification_status = "confirmed"
                    track.face_id = face_id
                    self.face_gallery.link_face_id(track.global_id, face_id)

                    if self.debug:
                        print(
                            f"[MTMC] Database verified: {cam_id}:T{track.track_id} G:{track.global_id} -> {face_id}"
                        )

    def _update_face_gallery(self, frame_num: int) -> None:
        """Update gallery with face vectors from tracked identities.

        Adds new face vectors to existing gallery entries for tracks that
        already have global_id assigned.
        """
        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                if track.global_id is None:
                    continue
                if track.face_vector is None:
                    continue

                # Add vector to gallery (will be throttled by min_vector_gap_frames)
                self.face_gallery.add_vector(
                    track.global_id,
                    track.face_vector,
                    frame_num,
                )

    def _propagate_across_cameras(self) -> None:
        """Propagate verification status across tracks with same global_id.

        When one track with global_id X is confirmed, all other tracks
        with global_id X should also be confirmed.
        """
        # Group tracks by global_id
        by_global_id: dict[int, list[BOTrack]] = {}
        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                if track.global_id is not None:
                    by_global_id.setdefault(track.global_id, []).append(track)

        # Propagate status within each group
        for global_id, tracks in by_global_id.items():
            # Find if any track is confirmed
            confirmed_track = next(
                (t for t in tracks if t.verification_status == "confirmed"), None
            )

            if confirmed_track:
                for track in tracks:
                    if track.verification_status != "confirmed":
                        track.verification_status = "confirmed"
                        track.face_id = confirmed_track.face_id

    def _check_alarms(self) -> None:
        """Escalate unverified tracks to alarm state after timeout."""
        for cam_id, tracker in self.trackers.items():
            for track in tracker.tracked_stracks:
                if track.verification_status == "waiting" and track.tracklet_len > self.alarm_timeout:
                    track.verification_status = "alarm"
                    if self.debug:
                        print(
                            f"[MTMC] Alarm: {cam_id}:T{track.track_id} G:{track.global_id} | "
                            f"frames={track.tracklet_len} (timeout={self.alarm_timeout})"
                        )

    def _cleanup_face_attempts(self) -> None:
        """Remove stale entries from face attempt tracking."""
        # Get all current track IDs
        current_track_ids = set()
        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                current_track_ids.add(track.track_id)

        # Remove entries for tracks no longer being tracked
        stale_ids = [tid for tid in self._last_face_attempt if tid not in current_track_ids]
        for tid in stale_ids:
            del self._last_face_attempt[tid]

    def verify_track(self, camera_id: str, track_id: int, face_id: str, frame_num: int = 0) -> bool:
        """Manually verify a track with an external face ID.

        Args:
            camera_id: Camera identifier
            track_id: Track ID to verify
            face_id: External face database ID (e.g., person name)
            frame_num: Current frame number

        Returns:
            True if track was found and verified, False otherwise
        """
        tracker = self.trackers.get(camera_id)
        if tracker:
            for track in tracker.tracked_stracks:
                if track.track_id == track_id:
                    if track.face_id and track.face_id != face_id:
                        print(f"[MTMC] Identity conflict: was '{track.face_id}', now '{face_id}'. Updating.")

                    track.verification_status = "confirmed"
                    track.face_id = face_id

                    # Update gallery with face_id link
                    if track.global_id is not None:
                        self.face_gallery.link_face_id(track.global_id, face_id)

                    return True
        return False

    def verify_by_global_id(self, global_id: int, face_id: str, frame_num: int = 0) -> int:
        """Verify all tracks with a given global_id.

        Useful when one camera identifies a person and you want to propagate
        the verification to all cameras tracking the same person.

        Args:
            global_id: Global ID to verify
            face_id: External face database ID
            frame_num: Current frame number

        Returns:
            Number of tracks verified
        """
        count = 0
        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                if track.global_id == global_id:
                    track.verification_status = "confirmed"
                    track.face_id = face_id
                    count += 1

        self.face_gallery.link_face_id(global_id, face_id)
        return count

    def get_tracks(self, camera_id: str | None = None) -> list[BOTrack]:
        """Get all tracks, optionally filtered by camera.

        Args:
            camera_id: Optional camera ID to filter by

        Returns:
            List of BOTrack objects
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
        """Get all tracks in alarm state."""
        return [t for t in self._get_all_tracks() if t.verification_status == "alarm"]

    def get_waiting(self) -> list[BOTrack]:
        """Get all tracks waiting for verification."""
        return [t for t in self._get_all_tracks() if t.verification_status == "waiting"]

    def get_confirmed(self) -> list[BOTrack]:
        """Get all confirmed (verified) tracks."""
        return [t for t in self._get_all_tracks() if t.verification_status == "confirmed"]

    def get_track_by_id(self, camera_id: str, track_id: int) -> BOTrack | None:
        """Get a specific track by camera and track ID."""
        tracker = self.trackers.get(camera_id)
        if tracker:
            for track in tracker.tracked_stracks:
                if track.track_id == track_id:
                    track.camera_id = camera_id
                    return track
        return None

    def get_tracks_by_global_id(self, global_id: int) -> list[BOTrack]:
        """Get all tracks with a specific global ID (across all cameras)."""
        return [t for t in self._get_all_tracks() if t.global_id == global_id]

    def get_known_identities(self) -> list[str]:
        """Get all known face IDs from the gallery."""
        return [
            identity.face_id
            for identity in self.face_gallery.identities.values()
            if identity.face_id is not None
        ]

    def get_identity_info(self, global_id: int) -> Optional[StoredFaceIdentity]:
        """Get identity info by global_id."""
        return self.face_gallery.get_identity(global_id)

    def reset(self) -> None:
        """Reset all trackers and clear the face gallery."""
        for tracker in self.trackers.values():
            tracker.reset()
        self.face_gallery.clear()
        self._last_face_attempt.clear()

    def __len__(self) -> int:
        return sum(len(tracker.tracked_stracks) for tracker in self.trackers.values())

    def __repr__(self) -> str:
        return (
            f"MTMCBridge(cameras={list(self.trackers.keys())}, "
            f"tracks={len(self)}, "
            f"gallery={len(self.face_gallery)}, "
            f"high_thresh={self.face_high_threshold}, "
            f"low_thresh={self.face_low_threshold})"
        )
