"""
catalog.py
----------
Enterprise Image Catalog and Metadata Management Engine.

Guarantees:
1. Complete CRUD operations for catalog entries and individual detected/manual objects.
2. Invariant: Manual corrections remain the immutable source of truth unless detection
   is explicitly requested with `force_rerun=True`.
3. Persistent storage with atomic JSON writes and asset archiving.
4. Clean separation of auto-generated metadata from user-editable fields.
"""

from __future__ import annotations
import os
import json
import shutil
import uuid
import datetime
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image

from models import (
    CatalogEntry,
    DetectedObject,
    BoundingBox,
    ColorInfo,
    DetectionOrigin,
    DetectionStatus,
    ImageCategory,
    CompositeMetadata
)
from detection import (
    HybridObjectDetector,
    COLOR_PALETTE,
    rgb_to_hex
)


class ImageCatalog:
    """
    Manages the reusable collection of images, detected objects, metadata records,
    and manual correction workflows.
    """

    def __init__(self, storage_dir: str = "catalog_storage"):
        self.storage_dir = os.path.abspath(storage_dir)
        self.images_dir = os.path.join(self.storage_dir, "images")
        self.cutouts_dir = os.path.join(self.storage_dir, "cutouts")
        self.metadata_file = os.path.join(self.storage_dir, "catalog_metadata.json")

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.cutouts_dir, exist_ok=True)

        self.detector = HybridObjectDetector(cutout_output_dir=self.cutouts_dir)
        self._entries: Dict[str, CatalogEntry] = {}  # id -> CatalogEntry
        self._name_index: Dict[str, str] = {}        # name -> id

        self.load_from_disk()

    # -------------------------------------------------------------------------
    # Core Entry Management
    # -------------------------------------------------------------------------

    def add_image(
        self,
        image_path: str,
        name: Optional[str] = None,
        category: ImageCategory = ImageCategory.UNKNOWN,
        run_detection: bool = True
    ) -> CatalogEntry:
        """
        Adds a new image into the catalog, copies the asset to storage,
        and optionally triggers automatic object detection.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Source image not found: {image_path}")

        entry_name = name or os.path.splitext(os.path.basename(image_path))[0]
        # Clean name
        entry_name = "".join(c for c in entry_name if c.isalnum() or c in ("_", "-"))

        entry_id = str(uuid.uuid4())[:8]

        # Inspect dimensions
        with Image.open(image_path) as im:
            width, height = im.size

        # Copy image asset to internal storage
        ext = os.path.splitext(image_path)[1].lower() or ".png"
        target_filename = f"{entry_name}_{entry_id}{ext}"
        stored_path = os.path.join(self.images_dir, target_filename)
        shutil.copy2(image_path, stored_path)

        entry = CatalogEntry(
            id=entry_id,
            name=entry_name,
            image_path=stored_path,
            width=width,
            height=height,
            category=category,
            status=DetectionStatus.PENDING,
            objects=[],
            has_manual_corrections=False,
            is_composite=False
        )

        self._entries[entry_id] = entry
        self._name_index[entry_name] = entry_id

        if run_detection:
            self.run_detection(entry_id, force_rerun=True)

        self.save_to_disk()
        return entry

    def register_composite(
        self,
        composite_image_path: str,
        name: str,
        composite_metadata: CompositeMetadata,
        run_detection: bool = True
    ) -> CatalogEntry:
        """
        Registers a newly generated composite image into the catalog as a full-fledged entry
        with associated composite provenance metadata.
        """
        entry = self.add_image(
            image_path=composite_image_path,
            name=name,
            category=ImageCategory.COMPOSITE,
            run_detection=run_detection
        )
        entry.is_composite = True
        entry.composite_metadata = composite_metadata
        self.save_to_disk()
        return entry

    # -------------------------------------------------------------------------
    # Detection & Source of Truth Invariant
    # -------------------------------------------------------------------------

    def run_detection(self, entry_id_or_name: str, force_rerun: bool = False) -> CatalogEntry:
        """
        Executes automatic object detection on a catalog image.

        SOURCE OF TRUTH INVARIANT:
        If an entry contains manual corrections (has_manual_corrections == True)
        and force_rerun is False, the manual corrections are preserved and detection
        is NOT rerun.
        """
        entry = self.get_entry(entry_id_or_name)
        if not entry:
            raise KeyError(f"Catalog entry not found: {entry_id_or_name}")

        if entry.has_manual_corrections and not force_rerun:
            # Preserving manual truth
            return entry

        detected_objects, inferred_category, status = self.detector.detect(
            image_path=entry.image_path,
            hint_category=entry.category,
            hint_name=entry.name
        )

        entry.objects = detected_objects
        entry.category = inferred_category
        entry.status = status
        entry.has_manual_corrections = False
        entry.update_timestamp()

        self.save_to_disk()
        return entry

    # -------------------------------------------------------------------------
    # Manual Correction Workflow (Add, Edit, Remove)
    # -------------------------------------------------------------------------

    def add_manual_object(
        self,
        entry_id_or_name: str,
        label: str,
        bbox: BoundingBox,
        color_name: str = "red",
        display_name: Optional[str] = None,
        confidence: float = 1.0,
        user_name: str = "annotator",
        notes: str = "Manually added object",
        generate_cutout: bool = True
    ) -> DetectedObject:
        """
        Scenario 2: Manually add an object when detection misses it.
        Locks entry.has_manual_corrections = True.
        """
        entry = self.get_entry(entry_id_or_name)
        if not entry:
            raise KeyError(f"Catalog entry not found: {entry_id_or_name}")

        # Bounding box bounds check against image dimensions
        bbox.x = max(0, min(bbox.x, entry.width - 1))
        bbox.y = max(0, min(bbox.y, entry.height - 1))
        bbox.width = min(bbox.width, entry.width - bbox.x)
        bbox.height = min(bbox.height, entry.height - bbox.y)

        rgb = COLOR_PALETTE.get(color_name.lower(), (128, 128, 128))
        hex_code = rgb_to_hex(rgb)
        color_info = ColorInfo(
            primary_name=color_name,
            primary_hex=hex_code,
            primary_rgb=rgb
        )

        cutout_path = None
        if generate_cutout and os.path.exists(entry.image_path):
            cutout_path = self._generate_cutout_from_bbox(entry.image_path, bbox, label)

        obj = DetectedObject.create_manual(
            label=label,
            bbox=bbox,
            color=color_info,
            display_name=display_name,
            confidence=confidence,
            user_name=user_name,
            notes=notes,
            cutout_mask_path=cutout_path
        )

        entry.add_object(obj)
        self.save_to_disk()
        return obj

    def edit_object(
        self,
        entry_id_or_name: str,
        object_id: str,
        new_label: Optional[str] = None,
        new_bbox: Optional[BoundingBox] = None,
        new_confidence: Optional[float] = None,
        new_color_name: Optional[str] = None,
        new_display_name: Optional[str] = None,
        user_name: str = "editor",
        notes: Optional[str] = None
    ) -> DetectedObject:
        """
        Scenario 3: Edit an object's name, color, confidence, or bounding box.
        Locks entry.has_manual_corrections = True.
        """
        entry = self.get_entry(entry_id_or_name)
        if not entry:
            raise KeyError(f"Catalog entry not found: {entry_id_or_name}")

        obj = entry.get_object(object_id)
        if not obj:
            raise KeyError(f"Object ID '{object_id}' not found in entry '{entry.name}'")

        new_color = None
        if new_color_name is not None:
            rgb = COLOR_PALETTE.get(new_color_name.lower(), (128, 128, 128))
            new_color = ColorInfo(
                primary_name=new_color_name,
                primary_hex=rgb_to_hex(rgb),
                primary_rgb=rgb
            )

        if new_bbox is not None:
            new_bbox.x = max(0, min(new_bbox.x, entry.width - 1))
            new_bbox.y = max(0, min(new_bbox.y, entry.height - 1))
            new_bbox.width = min(new_bbox.width, entry.width - new_bbox.x)
            new_bbox.height = min(new_bbox.height, entry.height - new_bbox.y)

        obj.apply_user_edit(
            new_label=new_label,
            new_bbox=new_bbox,
            new_confidence=new_confidence,
            new_color=new_color,
            new_display_name=new_display_name,
            user_name=user_name,
            notes=notes
        )

        # Regenerate cutout if bbox changed
        if new_bbox is not None and os.path.exists(entry.image_path):
            obj.cutout_mask_path = self._generate_cutout_from_bbox(
                entry.image_path, obj.bbox, obj.label
            )

        entry.has_manual_corrections = True
        entry.status = DetectionStatus.MANUALLY_EDITED
        entry.update_timestamp()

        self.save_to_disk()
        return obj

    def remove_object(self, entry_id_or_name: str, object_id: str) -> bool:
        """
        Deletes a detected or manual object from an entry.
        Locks entry.has_manual_corrections = True.
        """
        entry = self.get_entry(entry_id_or_name)
        if not entry:
            raise KeyError(f"Catalog entry not found: {entry_id_or_name}")

        removed = entry.remove_object(object_id)
        if removed:
            self.save_to_disk()
        return removed

    # -------------------------------------------------------------------------
    # Retrieval & Search Queries
    # -------------------------------------------------------------------------

    def get_entry(self, entry_id_or_name: str) -> Optional[CatalogEntry]:
        """Lookup by entry ID or friendly entry name."""
        if entry_id_or_name in self._entries:
            return self._entries[entry_id_or_name]
        if entry_id_or_name in self._name_index:
            return self._entries[self._name_index[entry_id_or_name]]
        return None

    def list_entries(
        self,
        category: Optional[ImageCategory] = None,
        status: Optional[DetectionStatus] = None
    ) -> List[CatalogEntry]:
        """Lists entries with optional filtering."""
        results = list(self._entries.values())
        if category:
            results = [e for e in results if e.category == category]
        if status:
            results = [e for e in results if e.status == status]
        return results

    def find_objects_by_label(self, label_query: str) -> List[Tuple[CatalogEntry, DetectedObject]]:
        """Finds all catalog objects matching a label query (case-insensitive substring)."""
        query = label_query.strip().lower()
        matches = []
        for entry in self._entries.values():
            for obj in entry.objects:
                if query in obj.label.lower() or query in obj.display_name.lower():
                    matches.append((entry, obj))
        return matches

    # -------------------------------------------------------------------------
    # Persistence & Serialization
    # -------------------------------------------------------------------------

    def save_to_disk(self):
        """Atomically saves catalog metadata to disk."""
        data = {
            "version": "2.0",
            "last_saved": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "entries_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries.values()]
        }
        temp_file = self.metadata_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        shutil.move(temp_file, self.metadata_file)

    def load_from_disk(self):
        """Loads catalog state from metadata file."""
        if not os.path.exists(self.metadata_file):
            return

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._entries.clear()
            self._name_index.clear()

            for item in data.get("entries", []):
                entry = CatalogEntry.from_dict(item)
                self._entries[entry.id] = entry
                self._name_index[entry.name] = entry.id
        except Exception as e:
            print(f"[Warning] Failed to load catalog metadata: {e}")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _generate_cutout_from_bbox(
        self,
        image_path: str,
        bbox: BoundingBox,
        label: str
    ) -> str:
        """Crops image region to create an alpha PNG cutout for downstream compositing."""
        with Image.open(image_path).convert("RGBA") as im:
            crop_box = (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
            cropped = im.crop(crop_box)

            # Check if image has white background, if so make it transparent
            data = cropped.getdata()
            new_data = []
            for item in data:
                # If pixel is near pure white, set alpha to 0
                if item[0] > 240 and item[1] > 240 and item[2] > 240:
                    new_data.append((item[0], item[1], item[2], 0))
                else:
                    new_data.append(item)
            cropped.putdata(new_data)

            clean_label = "".join(c for c in label if c.isalnum() or c == "_")
            filename = f"manual_{clean_label}_{bbox.x}_{bbox.y}.png"
            target = os.path.join(self.cutouts_dir, filename)
            cropped.save(target, "PNG")
            return os.path.abspath(target)
