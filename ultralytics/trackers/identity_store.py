# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Identity store for persistent reID-based identity matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import cosine


@dataclass
class StoredIdentity:
    face_id: str
    features: List[np.ndarray] = field(default_factory=list)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    global_ids: Set[int] = field(default_factory=set)


class IdentityStore:
    def __init__(
        self,
        max_features_per_identity: int = 10,
        match_threshold: float = 0.25,
        feature_ttl_frames: int = 108000,  # ~1 hour @ 30fps
    ):
        self.identities: Dict[str, StoredIdentity] = {}
        self.max_features_per_identity = max_features_per_identity
        self.match_threshold = match_threshold
        self.feature_ttl_frames = feature_ttl_frames

    def register(
        self,
        face_id: str,
        feature: np.ndarray,
        frame_num: int,
        global_id: Optional[int] = None,
    ) -> None:
        if feature is None:
            return

        feature = np.array(feature, dtype=np.float32)

        if face_id not in self.identities:
            self.identities[face_id] = StoredIdentity(
                face_id=face_id,
                features=[feature],
                first_seen_frame=frame_num,
                last_seen_frame=frame_num,
                global_ids={global_id} if global_id is not None else set(),
            )
        else:
            identity = self.identities[face_id]
            identity.last_seen_frame = frame_num

            if global_id is not None:
                identity.global_ids.add(global_id)

            # FIFO: keep last N features
            identity.features.append(feature)
            if len(identity.features) > self.max_features_per_identity:
                identity.features.pop(0)

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
            f"match_threshold={self.match_threshold})"
        )
