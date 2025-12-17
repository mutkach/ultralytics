# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import cdist, cosine


@dataclass
class StoredIdentity:
    face_id: str
    features: List[np.ndarray] = field(default_factory=list)
    feature_frames: List[int] = field(default_factory=list)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    global_ids: Set[int] = field(default_factory=set)


@dataclass
class StoredTrack:
    global_id: int
    features: List[np.ndarray] = field(default_factory=list)
    feature_frames: List[int] = field(default_factory=list)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    face_id: Optional[str] = None


class IdentityStore:
    def __init__(
        self,
        max_features_per_identity: int = 10,
        match_threshold: float = 0.25,
        feature_ttl_frames: int = 108000,
        min_feature_gap_frames: int = 30,
    ):
        self.identities: Dict[str, StoredIdentity] = {}
        self.max_features_per_identity = max_features_per_identity
        self.match_threshold = match_threshold
        self.feature_ttl_frames = feature_ttl_frames
        self.min_feature_gap_frames = min_feature_gap_frames

    def register(
        self,
        face_id: str,
        feature: np.ndarray,
        frame_num: int,
        global_id: Optional[int] = None,
    ) -> bool:
        if feature is None:
            return False

        feature = np.array(feature, dtype=np.float32)

        if face_id not in self.identities:
            self.identities[face_id] = StoredIdentity(
                face_id=face_id,
                features=[feature],
                feature_frames=[frame_num],
                first_seen_frame=frame_num,
                last_seen_frame=frame_num,
                global_ids={global_id} if global_id is not None else set(),
            )
            return True

        identity = self.identities[face_id]
        identity.last_seen_frame = frame_num

        if global_id is not None:
            identity.global_ids.add(global_id)

        if identity.feature_frames and frame_num - identity.feature_frames[-1] < self.min_feature_gap_frames:
            return False

        identity.features.append(feature)
        identity.feature_frames.append(frame_num)
        if len(identity.features) > self.max_features_per_identity:
            identity.features.pop(0)
            identity.feature_frames.pop(0)

        return True

    def match(self, query_feature: np.ndarray) -> Optional[Tuple[str, float]]:
        if query_feature is None or len(self.identities) == 0:
            return None

        query_feature = np.array(query_feature, dtype=np.float32)

        best_match = None
        best_distance = float("inf")

        for face_id, identity in self.identities.items():
            for stored_feat in identity.features:
                dist = cosine(query_feature, stored_feat)
                if dist < best_distance:
                    best_distance = dist
                    best_match = face_id

        if best_match is not None and best_distance < self.match_threshold:
            return (best_match, best_distance)

        return None

    def get_identity(self, face_id: str) -> Optional[StoredIdentity]:
        return self.identities.get(face_id)

    def get_all_identities(self) -> List[str]:
        return list(self.identities.keys())

    def cleanup(self, current_frame: int) -> int:
        stale_ids = [
            face_id
            for face_id, identity in self.identities.items()
            if current_frame - identity.last_seen_frame > self.feature_ttl_frames
        ]

        for face_id in stale_ids:
            del self.identities[face_id]

        return len(stale_ids)

    def clear(self) -> None:
        self.identities.clear()

    def __len__(self) -> int:
        return len(self.identities)

    def __repr__(self) -> str:
        total_features = sum(len(i.features) for i in self.identities.values())
        return (
            f"IdentityStore(identities={len(self.identities)}, "
            f"total_features={total_features}, "
            f"match_threshold={self.match_threshold}, "
            f"min_gap={self.min_feature_gap_frames})"
        )


class TrackGallery:
    def __init__(
        self,
        max_features_per_track: int = 10,
        match_threshold: float = 0.2,
        track_ttl_frames: int = 3600,
        min_feature_gap_frames: int = 15,
    ):
        self.tracks: Dict[int, StoredTrack] = {}
        self.max_features_per_track = max_features_per_track
        self.match_threshold = match_threshold
        self.track_ttl_frames = track_ttl_frames
        self.min_feature_gap_frames = min_feature_gap_frames

    def register(
        self,
        global_id: int,
        feature: np.ndarray,
        frame_num: int,
        face_id: Optional[str] = None,
    ) -> bool:
        if feature is None or global_id is None:
            return False

        feature = np.array(feature, dtype=np.float32)

        if global_id not in self.tracks:
            self.tracks[global_id] = StoredTrack(
                global_id=global_id,
                features=[feature],
                feature_frames=[frame_num],
                first_seen_frame=frame_num,
                last_seen_frame=frame_num,
                face_id=face_id,
            )
            return True

        track = self.tracks[global_id]
        track.last_seen_frame = frame_num

        if face_id and not track.face_id:
            track.face_id = face_id

        if track.feature_frames and frame_num - track.feature_frames[-1] < self.min_feature_gap_frames:
            return False

        track.features.append(feature)
        track.feature_frames.append(frame_num)
        if len(track.features) > self.max_features_per_track:
            track.features.pop(0)
            track.feature_frames.pop(0)

        return True

    def match(self, query_feature: np.ndarray) -> Optional[Tuple[int, float, Optional[str]]]:
        if query_feature is None or len(self.tracks) == 0:
            return None

        query_feature = np.array(query_feature, dtype=np.float32).reshape(1, -1)

        best_global_id = None
        best_distance = float("inf")
        best_face_id = None

        for global_id, track in self.tracks.items():
            if not track.features:
                continue
            gallery = np.vstack(track.features)
            distances = cdist(query_feature, gallery, metric='cosine').flatten()
            min_dist = distances.min()
            if min_dist < best_distance:
                best_distance = min_dist
                best_global_id = global_id
                best_face_id = track.face_id

        if best_global_id is not None and best_distance < self.match_threshold:
            return (best_global_id, best_distance, best_face_id)

        return None

    def update_face_id(self, global_id: int, face_id: str) -> bool:
        if global_id in self.tracks:
            self.tracks[global_id].face_id = face_id
            return True
        return False

    def merge(self, old_global_id: int, new_global_id: int) -> bool:
        if old_global_id not in self.tracks:
            return False
        if new_global_id not in self.tracks:
            self.tracks[new_global_id] = self.tracks.pop(old_global_id)
            self.tracks[new_global_id].global_id = new_global_id
            return True

        old_track = self.tracks.pop(old_global_id)
        new_track = self.tracks[new_global_id]

        for feat, frame in zip(old_track.features, old_track.feature_frames):
            if len(new_track.features) < self.max_features_per_track:
                new_track.features.append(feat)
                new_track.feature_frames.append(frame)

        if old_track.face_id and not new_track.face_id:
            new_track.face_id = old_track.face_id

        new_track.first_seen_frame = min(new_track.first_seen_frame, old_track.first_seen_frame)

        return True

    def cleanup(self, current_frame: int) -> int:
        stale_ids = [
            gid for gid, track in self.tracks.items()
            if current_frame - track.last_seen_frame > self.track_ttl_frames
        ]
        for gid in stale_ids:
            del self.tracks[gid]
        return len(stale_ids)

    def clear(self) -> None:
        self.tracks.clear()

    def __len__(self) -> int:
        return len(self.tracks)

    def __repr__(self) -> str:
        total_features = sum(len(t.features) for t in self.tracks.values())
        verified = sum(1 for t in self.tracks.values() if t.face_id)
        return (
            f"TrackGallery(tracks={len(self.tracks)}, verified={verified}, "
            f"total_features={total_features}, threshold={self.match_threshold})"
        )
