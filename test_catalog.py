"""
tests/test_catalog.py
---------------------
Formal Unit Tests for Image Catalog, Object Detection, Manual Corrections,
Source-of-Truth Invariants, and Natural-Language Compositing.
"""

import unittest
import os
import shutil
import tempfile

from models import (
    BoundingBox,
    ColorInfo,
    DetectedObject,
    DetectionOrigin,
    DetectionStatus,
    ImageCategory
)
from catalog import ImageCatalog
from compositor import CompositeGenerator
from asset_generator import generate_benchmark_assets


class TestImageCatalogSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="test_catalog_suite_")
        cls.assets_dir = os.path.join(cls.test_dir, "assets")
        cls.storage_dir = os.path.join(cls.test_dir, "storage")
        cls.generated_assets = generate_benchmark_assets(cls.assets_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.catalog = ImageCatalog(storage_dir=self.storage_dir)
        self.compositor = CompositeGenerator(self.catalog, output_dir=os.path.join(self.storage_dir, "composites"))

    def test_01_bounding_box_geometry(self):
        """Validates bounding box calculations, area, coordinates, and IoU."""
        box1 = BoundingBox(x=10, y=20, width=100, height=50)
        self.assertEqual(box1.x1, 10)
        self.assertEqual(box1.y1, 20)
        self.assertEqual(box1.x2, 110)
        self.assertEqual(box1.y2, 70)
        self.assertEqual(box1.area, 5000)

        box2 = BoundingBox(x=60, y=20, width=100, height=50)
        # Intersection: x in [60, 110], width=50, height=50 -> area=2500
        # Union: 5000 + 5000 - 2500 = 7500 -> IoU = 2500 / 7500 = 0.3333...
        iou = box1.iou(box2)
        self.assertAlmostEqual(iou, 0.3333333, places=4)

    def test_02_apple_and_orange_detection(self):
        """Verifies automatic detection on photographic fruit images."""
        apple_entry = self.catalog.add_image(self.generated_assets["apple"], name="apple", run_detection=True)
        self.assertEqual(apple_entry.status, DetectionStatus.DETECTED)
        self.assertEqual(len(apple_entry.objects), 1)
        self.assertEqual(apple_entry.objects[0].label, "apple")
        self.assertIn("red", apple_entry.objects[0].color.primary_name)

        orange_entry = self.catalog.add_image(self.generated_assets["orange"], name="orange", run_detection=True)
        self.assertEqual(len(orange_entry.objects), 1)
        self.assertEqual(orange_entry.objects[0].label, "orange")
        self.assertIn("orange", orange_entry.objects[0].color.primary_name)

    def test_03_cartoon_character_clothing_analysis(self):
        """Verifies cartoon figure detection and upper/lower clothing color extraction."""
        boy_entry = self.catalog.add_image(self.generated_assets["boy"], name="boy", run_detection=True)
        self.assertEqual(len(boy_entry.objects), 1)
        self.assertEqual(boy_entry.objects[0].label, "boy")
        clothing = boy_entry.objects[0].color.clothing_colors
        self.assertIsNotNone(clothing)
        self.assertIn("blue", clothing.get("top", "").lower())
        self.assertIn("khaki", clothing.get("bottom", "").lower())

    def test_04_manual_correction_source_of_truth_invariant(self):
        """
        Tests Scenario 2, 3, and 4:
        Manual edits must lock the entry and NEVER be overwritten unless force_rerun=True.
        """
        entry = self.catalog.get_entry("apple")
        orig_obj = entry.objects[0]
        orig_id = orig_obj.id

        # Edit object
        self.catalog.edit_object(
            entry_id_or_name="apple",
            object_id=orig_id,
            new_label="honeycrisp_apple",
            new_confidence=0.999,
            user_name="tester",
            notes="Variant taxonomy update"
        )
        updated_entry = self.catalog.get_entry("apple")
        self.assertTrue(updated_entry.has_manual_corrections)
        self.assertEqual(updated_entry.objects[0].label, "honeycrisp_apple")
        self.assertEqual(updated_entry.objects[0].origin, DetectionOrigin.USER_EDITED)

        # Safe rerun: should PRESERVE manual edits
        safe_entry = self.catalog.run_detection("apple", force_rerun=False)
        self.assertTrue(safe_entry.has_manual_corrections)
        self.assertEqual(safe_entry.objects[0].label, "honeycrisp_apple")

        # Explicit force rerun: should OVERWRITE manual edits
        reset_entry = self.catalog.run_detection("apple", force_rerun=True)
        self.assertFalse(reset_entry.has_manual_corrections)
        self.assertEqual(reset_entry.objects[0].label, "apple")
        self.assertEqual(reset_entry.objects[0].origin, DetectionOrigin.AUTO_DETECTED)

    def test_05_natural_language_composite_generation(self):
        """Verifies natural language composite generation using authentic catalog cutouts."""
        prompt = "Create an image containing one red apple and one orange"
        path, comp_entry = self.compositor.generate_from_prompt(
            prompt=prompt,
            target_name="unit_test_comp_fruits",
            enable_shadows=True
        )

        self.assertTrue(os.path.exists(path))
        self.assertTrue(comp_entry.is_composite)
        self.assertEqual(len(comp_entry.composite_metadata.source_assets), 2)
        # Verify provenance points to catalog assets
        source_names = [s.catalog_image_name for s in comp_entry.composite_metadata.source_assets]
        self.assertIn("apple", source_names)
        self.assertIn("orange", source_names)

    def test_06_empty_image_edge_case(self):
        """Verifies that empty/blank image yields 0 objects and status FAILED_NO_OBJECTS."""
        empty_entry = self.catalog.add_image(self.generated_assets["empty"], name="empty", run_detection=True)
        self.assertEqual(len(empty_entry.objects), 0)
        self.assertEqual(empty_entry.status, DetectionStatus.FAILED_NO_OBJECTS)


if __name__ == "__main__":
    unittest.main()
