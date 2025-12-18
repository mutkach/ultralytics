"""Simple crop storage for debugging track galleries."""

import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class CropStore:
    """Saves track crops to disk for debugging.

    Organizes crops by global_id:
        debug_crops/
            G1/
                f0001_cam1_T5.jpg
                f0045_cam2_T8.jpg
            G2/
                f0012_cam1_T3.jpg
    """

    def __init__(self, output_dir: str = "debug_crops"):
        self.output_dir = Path(output_dir)
        self.crop_count = 0

    def clear(self) -> None:
        """Clear all saved crops."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crop_count = 0
        print(f"[CropStore] Cleared {self.output_dir}")

    def save(
        self,
        crop: np.ndarray,
        global_id: int,
        frame_num: int,
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None,
        face_id: Optional[str] = None,
        is_rgb: bool = True,
    ) -> str:
        """Save a crop to disk.

        Args:
            crop: Image array (RGB or BGR format)
            global_id: Global track ID
            frame_num: Frame number
            camera_id: Camera identifier
            track_id: Local track ID
            face_id: Face ID if verified
            is_rgb: If True, convert RGB to BGR before saving

        Returns:
            Path to saved file
        """
        if crop is None or crop.size == 0:
            return ""

        # Create directory for this global_id
        gid_dir = self.output_dir / f"G{global_id}"
        gid_dir.mkdir(parents=True, exist_ok=True)

        # Build filename
        parts = [f"f{frame_num:05d}"]
        if camera_id:
            parts.append(camera_id)
        if track_id is not None:
            parts.append(f"T{track_id}")
        if face_id:
            # Sanitize face_id for filename
            safe_face_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in face_id)
            parts.append(safe_face_id)

        filename = "_".join(parts) + ".jpg"
        filepath = gid_dir / filename

        # Convert RGB to BGR for OpenCV saving
        if is_rgb:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # Save crop
        cv2.imwrite(str(filepath), crop)
        self.crop_count += 1

        return str(filepath)

    def __repr__(self) -> str:
        return f"CropStore(dir={self.output_dir}, crops={self.crop_count})"
