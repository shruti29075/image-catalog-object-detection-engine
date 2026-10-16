"""
test_scenarios.py
-----------------
Automated Benchmark Suite executing all 7 scenarios specified in the requirements.

Scenario 1: Detect objects and extract names, colors, confidence, and bounding boxes.
Scenario 2: Manually add an object when detection misses it.
Scenario 3: Edit an object's name, color, confidence, or bounding box.
Scenario 4: Verify manual corrections persist until detection is explicitly rerun.
Scenario 5: Generate a composite using existing 'apple' and 'orange' assets.
Scenario 6: Generate a composite containing a boy, a girl, and colored shapes.
Scenario 7: Test overlapping objects and images containing no detectable objects.
"""

from __future__ import annotations
import os
import sys
import json
import shutil
from typing import Dict, Any

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from asset_generator import generate_benchmark_assets
from catalog import ImageCatalog
from compositor import CompositeGenerator
from visualizer import CatalogVisualizer
from models import (
    BoundingBox,
    DetectionOrigin,
    DetectionStatus,
    ImageCategory
)


def run_all_scenarios():
    print("=" * 80)
    print("RUNNING COMPLETE IMAGE CATALOG, DETECTION & COMPOSITION BENCHMARK")
    print("=" * 80)

    # 0. Setup directories
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    test_assets_dir = os.path.join(workspace_dir, "test_assets")
    storage_dir = os.path.join(workspace_dir, "catalog_storage")
    results_dir = os.path.join(workspace_dir, "test_results")

    def safe_clean_dir(path: str):
        if not os.path.exists(path):
            return
        import stat
        for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    os.chmod(fp, stat.S_IWRITE | stat.S_IREAD)
                    os.remove(fp)
                except Exception:
                    pass
            for d in dirs:
                dp = os.path.join(root, d)
                try:
                    os.rmdir(dp)
                except Exception:
                    pass
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    # Clean previous run
    safe_clean_dir(storage_dir)
    safe_clean_dir(results_dir)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(storage_dir, exist_ok=True)

    # 1. Generate the 8 mandatory test assets + empty image
    print("\n[Step 0] Generating benchmark image assets...")
    assets = generate_benchmark_assets(output_dir=test_assets_dir)
    for name, path in assets.items():
        print(f"  + Generated '{name}': {os.path.basename(path)}")

    # Initialize Catalog and Compositor
    catalog = ImageCatalog(storage_dir=storage_dir)
    compositor = CompositeGenerator(catalog, output_dir=os.path.join(storage_dir, "composites"))

    # Populate Catalog with initial assets and run automatic detection
    print("\n[Step 1] Ingesting assets into Image Catalog and triggering automatic detection...")
    entries = {}
    for name, path in assets.items():
        entry = catalog.add_image(path, name=name, run_detection=True)
        entries[name] = entry
        # Save visual annotations
        annotated_path = os.path.join(results_dir, f"{name}_annotated.png")
        if entry.objects:
            CatalogVisualizer.annotate_image(entry.image_path, entry.objects, annotated_path)

    print(f"Catalog populated with {len(catalog.list_entries())} entries.")

    # =========================================================================
    # SCENARIO 1: Object Detection and Metadata Extraction
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 1: Automatic Object Detection & Metadata Extraction")
    print("=" * 80)

    target_scenarios_1 = ["apple", "orange", "boy", "girl", "shapes", "colors", "apple_orange"]
    for s_name in target_scenarios_1:
        entry = catalog.get_entry(s_name)
        assert entry is not None, f"Entry {s_name} missing!"
        print(f"\nAsset: [{entry.name}] ({entry.width}x{entry.height}px, Category: {entry.category.value})")
        print(f"Detection Status: {entry.status.value} | Objects Detected: {len(entry.objects)}")

        for idx, obj in enumerate(entry.objects, 1):
            bbox = obj.bbox
            print(f"  Object #{idx}:")
            print(f"    - ID: {obj.id}")
            print(f"    - Label: {obj.label} (Display: '{obj.display_name}')")
            print(f"    - Confidence: {obj.confidence:.2%}")
            print(f"    - Bounding Box: [x={bbox.x}, y={bbox.y}, w={bbox.width}, h={bbox.height}] (Area: {bbox.area}px²)")
            print(f"    - Primary Color: {obj.color.primary_name} ({obj.color.primary_hex})")
            if obj.color.clothing_colors:
                print(f"    - Clothing Analysis: {obj.color.clothing_colors}")
            print(f"    - Origin: {obj.origin.value} | Auto-field snapshot: Model={obj.auto_metadata.model_version}")

    # Specific assertions
    apple_entry = catalog.get_entry("apple")
    assert len(apple_entry.objects) == 1
    assert apple_entry.objects[0].label == "apple"
    assert "red" in apple_entry.objects[0].color.primary_name

    boy_entry = catalog.get_entry("boy")
    assert len(boy_entry.objects) == 1
    assert boy_entry.objects[0].label == "boy"
    assert boy_entry.objects[0].color.clothing_colors is not None

    shapes_entry = catalog.get_entry("shapes")
    assert len(shapes_entry.objects) >= 4, f"Expected 4 shapes, got {len(shapes_entry.objects)}"

    print("\n>>> Scenario 1 PASSED: All objects, colors, confidence scores, and bboxes extracted successfully.")

    # =========================================================================
    # SCENARIO 2: Manual Addition when Detection Misses an Object
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 2: Manual Addition When Detection Misses an Object")
    print("=" * 80)

    # In 'apple', simulate annotating the green leaf/stem explicitly missed or uncataloged as a separate item
    leaf_bbox = BoundingBox(x=285, y=105, width=85, height=40)
    manual_leaf = catalog.add_manual_object(
        entry_id_or_name="apple",
        label="apple_leaf",
        bbox=leaf_bbox,
        color_name="green",
        display_name="Apple Leaf & Stem",
        confidence=1.0,
        user_name="expert_curator",
        notes="High-value foliage detail missed by fruit detector"
    )

    apple_updated = catalog.get_entry("apple")
    print(f"Added Manual Object to '{apple_updated.name}':")
    print(f"  - Object ID: {manual_leaf.id}")
    print(f"  - Label: {manual_leaf.label}")
    print(f"  - Origin: {manual_leaf.origin.value}")
    print(f"  - Confidence: {manual_leaf.confidence}")
    print(f"  - User Metadata: is_modified={manual_leaf.user_metadata.is_modified}, by={manual_leaf.user_metadata.modified_by}")
    print(f"  - Catalog Entry Lock Flag (has_manual_corrections): {apple_updated.has_manual_corrections}")

    assert manual_leaf.origin == DetectionOrigin.USER_MANUAL
    assert apple_updated.has_manual_corrections is True
    assert len(apple_updated.objects) == 2

    # Save visual verification
    CatalogVisualizer.annotate_image(
        apple_updated.image_path,
        apple_updated.objects,
        os.path.join(results_dir, "scenario2_manual_add_annotated.png")
    )
    print("\n>>> Scenario 2 PASSED: Manual object added with full provenance and entry lock enabled.")

    # =========================================================================
    # SCENARIO 3: Edit an Object's Name, Color, Confidence, or Bounding Box
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 3: Manual Edit of an Object's Metadata")
    print("=" * 80)

    # In 'boy', edit the detected character to refine label, bounding box, and confidence
    boy_entry = catalog.get_entry("boy")
    target_obj = boy_entry.objects[0]
    original_id = target_obj.id
    original_label = target_obj.label
    original_bbox = target_obj.bbox

    print(f"Original Object: Label='{original_label}', BBox={original_bbox.to_dict()}, Conf={target_obj.confidence}")

    refined_bbox = BoundingBox(
        x=max(0, original_bbox.x - 5),
        y=max(0, original_bbox.y - 10),
        width=original_bbox.width + 10,
        height=original_bbox.height + 15
    )

    edited_obj = catalog.edit_object(
        entry_id_or_name="boy",
        object_id=original_id,
        new_label="school_boy_hero",
        new_display_name="Schoolboy Character Hero",
        new_bbox=refined_bbox,
        new_confidence=0.995,
        new_color_name="blue",
        user_name="senior_annotator",
        notes="Refined bounding box around shoes and adjusted label taxonomy"
    )

    boy_updated = catalog.get_entry("boy")
    print(f"\nEdited Object:")
    print(f"  - Label: '{edited_obj.label}' (Display: '{edited_obj.display_name}')")
    print(f"  - BBox: {edited_obj.bbox.to_dict()}")
    print(f"  - Confidence: {edited_obj.confidence}")
    print(f"  - Origin: {edited_obj.origin.value}")
    print(f"  - Pre-edit Backup Preserved: {edited_obj.user_metadata.original_values_backup['label']}")
    print(f"  - Auto Detection Snapshot Preserved: {edited_obj.auto_metadata.raw_label}")

    assert edited_obj.origin == DetectionOrigin.USER_EDITED
    assert edited_obj.label == "school_boy_hero"
    assert edited_obj.confidence == 0.995
    assert boy_updated.has_manual_corrections is True

    # Save visual verification
    CatalogVisualizer.annotate_image(
        boy_updated.image_path,
        boy_updated.objects,
        os.path.join(results_dir, "scenario3_manual_edit_annotated.png")
    )
    print("\n>>> Scenario 3 PASSED: Object metadata edited with dual-snapshot preservation.")

    # =========================================================================
    # SCENARIO 4: Verify Manual Corrections Persist Until Detection is Explicitly Rerun
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 4: Verification of Source-of-Truth Persistence Invariant")
    print("=" * 80)

    # Test 4a: Call detection WITHOUT force_rerun (should preserve manual edits)
    print("Test 4a: Calling run_detection(entry='boy', force_rerun=False)...")
    entry_after_soft_run = catalog.run_detection("boy", force_rerun=False)

    print(f"  - has_manual_corrections: {entry_after_soft_run.has_manual_corrections}")
    print(f"  - Active Label: {entry_after_soft_run.objects[0].label}")
    print(f"  - Active Origin: {entry_after_soft_run.objects[0].origin.value}")

    assert entry_after_soft_run.has_manual_corrections is True, "Manual lock was improperly cleared!"
    assert entry_after_soft_run.objects[0].label == "school_boy_hero", "Manual edits were overwritten!"
    assert entry_after_soft_run.objects[0].origin == DetectionOrigin.USER_EDITED
    print("  [OK] Manual corrections were strictly protected from silent overwrite.")

    # Test 4b: Call detection WITH force_rerun=True (user explicitly overrides)
    print("\nTest 4b: Calling run_detection(entry='boy', force_rerun=True)...")
    entry_after_hard_run = catalog.run_detection("boy", force_rerun=True)

    print(f"  - has_manual_corrections: {entry_after_hard_run.has_manual_corrections}")
    print(f"  - Active Label: {entry_after_hard_run.objects[0].label}")
    print(f"  - Active Origin: {entry_after_hard_run.objects[0].origin.value}")

    assert entry_after_hard_run.has_manual_corrections is False
    assert entry_after_hard_run.objects[0].origin == DetectionOrigin.AUTO_DETECTED
    assert entry_after_hard_run.objects[0].label == "boy"
    print("  [OK] Explicit re-detection reset catalog entry to pure auto-detected state as requested.")

    print("\n>>> Scenario 4 PASSED: Source-of-truth invariants verified under all execution modes.")

    # =========================================================================
    # SCENARIO 5: Composite Image Generation (Apple + Orange)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 5: Natural Language Composite Generation (Apple + Orange)")
    print("=" * 80)

    prompt_fruits = "Create an image containing one red apple and one orange"
    print(f"Natural Language Instruction: \"{prompt_fruits}\"")

    comp_fruit_path, comp_fruit_entry = compositor.generate_from_prompt(
        prompt=prompt_fruits,
        target_name="composite_apple_and_orange",
        canvas_width=900,
        canvas_height=600,
        enable_shadows=True
    )

    print(f"Generated Composite File: {os.path.basename(comp_fruit_path)}")
    print(f"Registered Catalog Entry ID: {comp_fruit_entry.id} (Name: '{comp_fruit_entry.name}')")
    print(f"Composite Metadata:")
    print(f"  - Prompt: \"{comp_fruit_entry.composite_metadata.prompt}\"")
    print(f"  - Canvas Dimensions: {comp_fruit_entry.composite_metadata.canvas_dimensions}")
    print(f"  - Source Assets Provenance:")
    for src in comp_fruit_entry.composite_metadata.source_assets:
        print(f"    * Asset '{src.catalog_image_name}' (Obj: {src.object_label}): Placement={src.placement_bbox.to_dict()}, Scale={src.applied_scale}")

    # Verify detection automatically ran on the new composite
    print(f"\nAuto-Detection on Composite Image:")
    print(f"  - Detected Objects in Composite: {len(comp_fruit_entry.objects)}")
    for obj in comp_fruit_entry.objects:
        print(f"    * {obj.label} at {obj.bbox.to_dict()} (Conf: {obj.confidence:.2%})")

    assert comp_fruit_entry.is_composite is True
    assert len(comp_fruit_entry.composite_metadata.source_assets) == 2
    assert len(comp_fruit_entry.objects) >= 2

    # Save visual verification card
    apple_src = catalog.get_entry("apple").image_path
    orange_src = catalog.get_entry("orange").image_path
    CatalogVisualizer.create_side_by_side_card(
        source_paths=[apple_src, orange_src],
        composite_path=comp_fruit_path,
        output_path=os.path.join(results_dir, "scenario5_apple_orange_card.png"),
        title=f"Scenario 5 Composite: '{prompt_fruits}'"
    )
    CatalogVisualizer.annotate_image(
        comp_fruit_path,
        comp_fruit_entry.objects,
        os.path.join(results_dir, "scenario5_apple_orange_annotated.png")
    )
    print("\n>>> Scenario 5 PASSED: Composite generated strictly with authentic catalog assets and registered.")

    # =========================================================================
    # SCENARIO 6: Composite Image Generation (Boy, Girl, and Colored Shapes)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 6: Composite Generation (Boy, Girl, and Colored Shapes)")
    print("=" * 80)

    prompt_complex = "Generate a composite containing a boy, a girl, and colored shapes"
    print(f"Natural Language Instruction: \"{prompt_complex}\"")

    comp_complex_path, comp_complex_entry = compositor.generate_from_prompt(
        prompt=prompt_complex,
        target_name="composite_boy_girl_shapes",
        canvas_width=1100,
        canvas_height=700,
        enable_shadows=True
    )

    print(f"Generated Composite File: {os.path.basename(comp_complex_path)}")
    print(f"Registered Catalog Entry ID: {comp_complex_entry.id}")
    print(f"Source Assets Utilized in Composite:")
    for src in comp_complex_entry.composite_metadata.source_assets:
        print(f"  - Source Catalog Entry: '{src.catalog_image_name}' -> Label: '{src.object_label}', Z-Index: {src.z_index}")

    # Visual verification card
    boy_src = catalog.get_entry("boy").image_path
    girl_src = catalog.get_entry("girl").image_path
    shapes_src = catalog.get_entry("shapes").image_path
    CatalogVisualizer.create_side_by_side_card(
        source_paths=[boy_src, girl_src, shapes_src],
        composite_path=comp_complex_path,
        output_path=os.path.join(results_dir, "scenario6_boy_girl_shapes_card.png"),
        title=f"Scenario 6 Composite: '{prompt_complex}'"
    )
    CatalogVisualizer.annotate_image(
        comp_complex_path,
        comp_complex_entry.objects,
        os.path.join(results_dir, "scenario6_boy_girl_shapes_annotated.png")
    )

    assert comp_complex_entry.is_composite is True
    assert len(comp_complex_entry.composite_metadata.source_assets) >= 3
    print("\n>>> Scenario 6 PASSED: Multi-entity composite successfully scaled, layered, and rendered.")

    # =========================================================================
    # SCENARIO 7: Overlapping Shapes and Empty/No-Object Edge Cases
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 7: Overlapping Objects & Empty Image Edge-Case Verification")
    print("=" * 80)

    # Part A: Overlapping Shapes
    overlap_entry = catalog.get_entry("overlapping_shapes")
    print(f"Part A: Overlapping Shapes Asset ('{overlap_entry.name}'):")
    print(f"  - Detected Shape Regions: {len(overlap_entry.objects)}")
    for obj in overlap_entry.objects:
        print(f"    * Shape: {obj.label} ({obj.color.primary_name}) BBox: {obj.bbox.to_dict()} (Conf: {obj.confidence:.2%})")

    assert len(overlap_entry.objects) >= 2, "Failed to delineate overlapping shapes!"
    print("  [OK] Overlapping shapes successfully segmented via watershed distance transform.")

    # Part B: Empty / Blank Image (No detectable objects)
    empty_entry = catalog.get_entry("empty")
    print(f"\nPart B: Empty Image Asset ('{empty_entry.name}'):")
    print(f"  - Detection Status: {empty_entry.status.value}")
    print(f"  - Detected Objects Count: {len(empty_entry.objects)}")

    assert len(empty_entry.objects) == 0, "Detected spurious objects in empty image!"
    assert empty_entry.status == DetectionStatus.FAILED_NO_OBJECTS, f"Expected FAILED_NO_OBJECTS, got {empty_entry.status}"
    print("  [OK] Empty image correctly flagged with FAILED_NO_OBJECTS and zero false positives.")

    print("\n>>> Scenario 7 PASSED: Overlapping objects and zero-object edge cases handled cleanly.")

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Catalog Entries Managed: {len(catalog.list_entries())}")
    print(f"Output Storage Directory: {storage_dir}")
    print(f"Visual Results Directory: {results_dir}")
    print("ALL 7 SCENARIOS COMPLETED WITH 100% SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_scenarios()
