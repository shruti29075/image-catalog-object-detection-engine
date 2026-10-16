"""
app.py
------
Interactive Visual Web Dashboard for the Image Catalog, Detection,
Manual Correction, and Composite Generation OS.

Built with Streamlit and modern visual design principles.
"""

from __future__ import annotations
import os
import json
import streamlit as st
from PIL import Image

# Core System Imports
from models import (
    BoundingBox,
    DetectionOrigin,
    DetectionStatus,
    ImageCategory
)
from catalog import ImageCatalog
from compositor import CompositeGenerator
from visualizer import CatalogVisualizer
from asset_generator import generate_benchmark_assets

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionStudio | Image Catalog & Compositor OS",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E293B 0%, #3B82F6 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #64748B;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .badge-auto {
        background: #DCFCE7;
        color: #15803D;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-manual {
        background: #DBEAFE;
        color: #1D4ED8;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-edited {
        background: #FFEDD5;
        color: #C2410C;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_catalog_system():
    workspace = os.path.dirname(os.path.abspath(__file__))
    test_assets = os.path.join(workspace, "test_assets")
    storage = os.path.join(workspace, "catalog_storage")

    # Generate benchmark assets if absent
    if not os.path.exists(test_assets) or len(os.listdir(test_assets)) < 8:
        generate_benchmark_assets(test_assets)

    catalog = ImageCatalog(storage_dir=storage)

    # Ingest benchmark assets if catalog is fresh
    if len(catalog.list_entries()) == 0:
        for fname in sorted(os.listdir(test_assets)):
            if fname.endswith(".png"):
                fpath = os.path.join(test_assets, fname)
                bname = os.path.splitext(fname)[0]
                catalog.add_image(fpath, name=bname, run_detection=True)

    compositor = CompositeGenerator(catalog, output_dir=os.path.join(storage, "composites"))
    return catalog, compositor


catalog, compositor = get_catalog_system()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **VisionStudio OS**")
    st.caption("Image Catalog, Detection & Composition Engine")

    nav = st.radio(
        "Navigation",
        [
            "🖼️ Image Catalog Gallery",
            "🔍 Object Detection & Correction",
            "🎨 Natural Language Compositor",
            "📊 Architecture & Diagnostics"
        ]
    )

    st.divider()
    entries = catalog.list_entries()
    total_objs = sum(len(e.objects) for e in entries)
    manual_entries = sum(1 for e in entries if e.has_manual_corrections)
    comp_entries = sum(1 for e in entries if e.is_composite)

    st.markdown("**Catalog Overview:**")
    st.markdown(f"- 📁 Total Images: **{len(entries)}**")
    st.markdown(f"- 🏷️ Detected Objects: **{total_objs}**")
    st.markdown(f"- 🔒 Manually Corrected: **{manual_entries}**")
    st.markdown(f"- 🎨 Composites Created: **{comp_entries}**")

    st.divider()
    if st.button("🔄 Reload Catalog from Storage", use_container_width=True):
        catalog.load_from_disk()
        st.toast("Catalog refreshed from persistent storage!", icon="✅")
        st.rerun()


# -----------------------------------------------------------------------------
# TAB 1: Image Catalog Gallery
# -----------------------------------------------------------------------------
if nav == "🖼️ Image Catalog Gallery":
    st.markdown('<div class="main-title">Image Catalog Directory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Browse persistent image assets, metadata schemas, and detected features.</div>', unsafe_allow_html=True)

    # Quick Filters
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        cat_filter = st.selectbox(
            "Filter by Category",
            ["All"] + [c.value for c in ImageCategory]
        )
    with col_f2:
        search_query = st.text_input("Search by Object or Image Name", "")

    all_entries = catalog.list_entries()
    filtered = all_entries
    if cat_filter != "All":
        filtered = [e for e in filtered if e.category.value == cat_filter]
    if search_query:
        sq = search_query.lower()
        filtered = [
            e for e in filtered
            if sq in e.name.lower() or any(sq in o.label.lower() for o in e.objects)
        ]

    st.markdown(f"**Showing {len(filtered)} catalog entries:**")

    # Render Grid of Cards
    cols = st.columns(3)
    for idx, entry in enumerate(filtered):
        with cols[idx % 3]:
            with st.container(border=True):
                # Thumbnail
                if os.path.exists(entry.image_path):
                    st.image(entry.image_path, use_container_width=True)

                st.markdown(f"**{entry.name}**")
                # Badges
                badges = []
                if entry.is_composite:
                    badges.append("<span class='badge-edited'>COMPOSITE</span>")
                if entry.has_manual_corrections:
                    badges.append("<span class='badge-manual'>MANUAL EDITS</span>")
                else:
                    badges.append("<span class='badge-auto'>AUTO VERIFIED</span>")

                st.markdown(" ".join(badges), unsafe_allow_html=True)
                st.caption(f"Dim: {entry.width}x{entry.height}px | Status: `{entry.status.value}`")

                # Objects list
                if entry.objects:
                    obj_tags = [f"`{o.label}` ({o.color.primary_name})" for o in entry.objects]
                    st.markdown(f"**Objects ({len(entry.objects)}):** " + ", ".join(obj_tags[:3]))
                else:
                    st.caption("No objects detected (Empty image)")

                # Expandable JSON metadata
                with st.expander("View Metadata JSON"):
                    st.json(entry.to_dict())


# -----------------------------------------------------------------------------
# TAB 2: Object Detection & Manual Correction (Scenarios 1, 2, 3, 4)
# -----------------------------------------------------------------------------
elif nav == "🔍 Object Detection & Correction":
    st.markdown('<div class="main-title">Object Detection & Manual Correction Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect automated bounding boxes, manually add/edit objects, and enforce Source-of-Truth persistence.</div>', unsafe_allow_html=True)

    entries = catalog.list_entries()
    entry_names = [e.name for e in entries]

    selected_name = st.selectbox("Select Catalog Image to Inspect / Edit:", entry_names, index=0)
    entry = catalog.get_entry(selected_name)

    col_view1, col_view2 = st.columns(2)

    with col_view1:
        st.markdown("**Original Image Asset:**")
        if os.path.exists(entry.image_path):
            st.image(entry.image_path, use_container_width=True)
        st.caption(f"Path: `{entry.image_path}` | Res: {entry.width}x{entry.height}px")

    with col_view2:
        st.markdown("**Visual Annotated Detections:**")
        annotated_cache = os.path.join(
            catalog.storage_dir, f"{entry.name}_preview_annotated.png"
        )
        CatalogVisualizer.annotate_image(entry.image_path, entry.objects, annotated_cache)
        st.image(annotated_cache, use_container_width=True)
        st.markdown(
            "**Legend:** 🟢 `<span class='badge-auto'>Auto-Detected</span>` | "
            "🔵 `<span class='badge-manual'>Manually Added</span>` | "
            "🟠 `<span class='badge-edited'>User Edited</span>`",
            unsafe_allow_html=True
        )

    st.divider()

    # Objects Table
    st.markdown(f"### Detected Objects in `{entry.name}` ({len(entry.objects)})")
    if entry.objects:
        obj_data = []
        for o in entry.objects:
            obj_data.append({
                "Object ID": o.id,
                "Label": o.label,
                "Display Name": o.display_name,
                "Confidence": f"{o.confidence:.1%}",
                "Bounding Box [x, y, w, h]": f"[{o.bbox.x}, {o.bbox.y}, {o.bbox.width}, {o.bbox.height}]",
                "Color": f"{o.color.primary_name} ({o.color.primary_hex})",
                "Origin": o.origin.value,
                "User Modified": o.user_metadata.is_modified,
                "Clothing": str(o.color.clothing_colors or "-")
            })
        st.dataframe(obj_data, use_container_width=True)
    else:
        st.info("No detected objects in this entry (Empty image scenario).")

    # Management Forms
    st.markdown("---")
    tab_add, tab_edit, tab_detect = st.tabs([
        "➕ Scenario 2: Manually Add Missed Object",
        "✏️ Scenario 3: Edit Detected Object",
        "🔄 Scenario 4: Detection & Source-of-Truth Invariant"
    ])

    # Tab Add
    with tab_add:
        st.markdown("**Add an object that automatic detection missed:**")
        with st.form("add_manual_object_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_lbl = st.text_input("Object Label", value="special_feature")
                new_disp = st.text_input("Display Name", value="Special Feature")
            with c2:
                new_color = st.selectbox("Primary Color", ["red", "green", "blue", "yellow", "orange", "purple", "white", "black"])
                new_conf = st.slider("Confidence Score", 0.0, 1.0, 1.0, 0.05)
            with c3:
                user_nm = st.text_input("Curator / Annotator Name", value="Curator_1")
                notes_text = st.text_input("Annotation Notes", value="Missed by baseline detector")

            st.markdown("**Bounding Box Coordinates (pixels):**")
            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            with b_c1:
                bx = st.number_input("x", 0, entry.width, 50)
            with b_c2:
                by = st.number_input("y", 0, entry.height, 50)
            with b_c3:
                bw = st.number_input("width", 10, entry.width, 100)
            with b_c4:
                bh = st.number_input("height", 10, entry.height, 100)

            if st.form_submit_button("Add Manual Object to Catalog", use_container_width=True):
                added_obj = catalog.add_manual_object(
                    entry_id_or_name=entry.name,
                    label=new_lbl,
                    bbox=BoundingBox(int(bx), int(by), int(bw), int(bh)),
                    color_name=new_color,
                    display_name=new_disp,
                    confidence=new_conf,
                    user_name=user_nm,
                    notes=notes_text
                )
                st.success(f"Added manual object '{added_obj.label}' (ID: {added_obj.id}). Manual corrections locked!")
                st.rerun()

    # Tab Edit
    with tab_edit:
        st.markdown("**Edit an existing object's label, bounding box, color, or confidence:**")
        if entry.objects:
            obj_choices = {f"{o.id} - {o.label} ({o.origin.value})": o.id for o in entry.objects}
            selected_obj_key = st.selectbox("Select Object to Edit", list(obj_choices.keys()))
            target_obj_id = obj_choices[selected_obj_key]
            curr_obj = entry.get_object(target_obj_id)

            with st.form("edit_object_form"):
                e_c1, e_c2, e_c3 = st.columns(3)
                with e_c1:
                    edit_lbl = st.text_input("New Label", value=curr_obj.label)
                    edit_disp = st.text_input("New Display Title", value=curr_obj.display_name)
                with e_c2:
                    edit_col = st.selectbox("Color", ["red", "green", "blue", "yellow", "orange", "purple", "white", "black"], index=0)
                    edit_conf = st.slider("Confidence", 0.0, 1.0, float(curr_obj.confidence), 0.01)
                with e_c3:
                    edit_user = st.text_input("Editor Name", value="Senior_Editor")
                    edit_notes = st.text_input("Correction Rationale", value="Corrected boundary and taxonomy")

                eb_c1, eb_c2, eb_c3, eb_c4 = st.columns(4)
                with eb_c1:
                    ebx = st.number_input("x", 0, entry.width, curr_obj.bbox.x)
                with eb_c2:
                    eby = st.number_input("y", 0, entry.height, curr_obj.bbox.y)
                with eb_c3:
                    ebw = st.number_input("width", 5, entry.width, curr_obj.bbox.width)
                with eb_c4:
                    ebh = st.number_input("height", 5, entry.height, curr_obj.bbox.height)

                col_btn1, col_btn2 = st.columns([3, 1])
                with col_btn1:
                    save_edit = st.form_submit_button("Save Corrections", use_container_width=True)
                with col_btn2:
                    delete_obj = st.form_submit_button("🗑️ Delete Object", use_container_width=True)

                if save_edit:
                    catalog.edit_object(
                        entry_id_or_name=entry.name,
                        object_id=curr_obj.id,
                        new_label=edit_lbl,
                        new_display_name=edit_disp,
                        new_color_name=edit_col,
                        new_confidence=edit_conf,
                        new_bbox=BoundingBox(int(ebx), int(eby), int(ebw), int(ebh)),
                        user_name=edit_user,
                        notes=edit_notes
                    )
                    st.success(f"Object '{curr_obj.id}' successfully updated! Invariant locked.")
                    st.rerun()

                if delete_obj:
                    catalog.remove_object(entry.name, curr_obj.id)
                    st.warning(f"Object '{curr_obj.id}' removed from catalog.")
                    st.rerun()
        else:
            st.info("No objects present to edit.")

    # Tab Detect / Persistence Invariant
    with tab_detect:
        st.markdown("**Source of Truth & Re-Detection Controls:**")
        st.markdown(f"- Entry Lock Status: **`{'LOCKED (has_manual_corrections=True)' if entry.has_manual_corrections else 'CLEAN (Pure Auto-Detection)'}`**")

        c_d1, c_d2 = st.columns(2)
        with c_d1:
            st.markdown("#### Safe Re-run (Default)")
            st.caption("Preserves all manual additions, edits, and deletions. Will NOT overwrite user corrections.")
            if st.button("Run Detection (Safe / Respect Manual)", use_container_width=True):
                updated = catalog.run_detection(entry.name, force_rerun=False)
                if updated.has_manual_corrections:
                    st.info("Manual corrections were preserved! Automatic detection was safely bypassed.")
                else:
                    st.success("Detection completed.")
                st.rerun()

        with c_d2:
            st.markdown("#### Explicit Force Re-run")
            st.caption("WARNING: Explicit user request to re-run detection and overwrite all manual corrections with pure automated results.")
            if st.button("⚠️ Force Rerun Detection (Reset to Auto)", use_container_width=True):
                updated = catalog.run_detection(entry.name, force_rerun=True)
                st.warning("Entry reset to pure automated detections. Manual corrections cleared.")
                st.rerun()


# -----------------------------------------------------------------------------
# TAB 3: Natural Language Compositor (Scenarios 5, 6)
# -----------------------------------------------------------------------------
elif nav == "🎨 Natural Language Compositor":
    st.markdown('<div class="main-title">Natural Language Composite Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Generate brand-new scenes assembled strictly from authentic catalog image cutouts. No generative replacements.</div>', unsafe_allow_html=True)

    col_input, col_settings = st.columns([3, 2])

    with col_input:
        st.markdown("**Natural Language Composition Prompt:**")
        preset = st.selectbox(
            "Select Prompt Preset or Type Below:",
            [
                "Create an image containing one red apple and one orange",
                "Generate a composite containing a boy, a girl, and colored shapes",
                "Create a scene with a cartoon boy and one red apple",
                "Create an image containing a cartoon girl and colored shapes",
                "Custom Prompt..."
            ]
        )

        if preset != "Custom Prompt...":
            default_prompt = preset
        else:
            default_prompt = ""

        prompt_text = st.text_area("Composition Instruction:", value=default_prompt, height=90)

    with col_settings:
        st.markdown("**Canvas & Rendering Parameters:**")
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            cw = st.slider("Canvas Width", 700, 1200, 950, 50)
            enable_sh = st.checkbox("Contact Drop-Shadows", value=True)
        with s_c2:
            ch = st.slider("Canvas Height", 400, 800, 620, 20)
            bg_style = st.selectbox("Background Style", ["studio_clean", "pure_white"])

        custom_name = st.text_input("Composite Catalog Entry Name:", value="my_composite_scene")

    if st.button("✨ Generate Composite Image From Catalog Assets", use_container_width=True, type="primary"):
        if not prompt_text.strip():
            st.error("Please enter a composition prompt.")
        else:
            with st.spinner("Resolving catalog cutouts, calculating layout, and rendering composite..."):
                try:
                    comp_path, comp_entry = compositor.generate_from_prompt(
                        prompt=prompt_text,
                        target_name=custom_name,
                        canvas_width=cw,
                        canvas_height=ch,
                        background_style=bg_style,
                        enable_shadows=enable_sh,
                        auto_register_in_catalog=True
                    )

                    st.success(f"Composite generated and registered in catalog as '{comp_entry.name}' (ID: {comp_entry.id})!")

                    # Show Results
                    st.divider()
                    st.markdown("### Generated Composite Result")

                    c_res1, c_res2 = st.columns([3, 2])
                    with c_res1:
                        st.image(comp_path, use_container_width=True)
                        st.caption(f"Composite Output: `{comp_path}`")

                    with c_res2:
                        st.markdown("**Asset Provenance & Layout Plan:**")
                        st.caption(f"Prompt: *\"{comp_entry.composite_metadata.prompt}\"*")
                        for src in comp_entry.composite_metadata.source_assets:
                            st.markdown(
                                f"- **Source Entry:** `{src.catalog_image_name}` (Obj: `{src.object_label}`)\n"
                                f"  - Scale: `{src.applied_scale}x` | Z-Index: `{src.z_index}`\n"
                                f"  - Placed BBox: `{src.placement_bbox.to_dict()}`"
                            )

                        st.markdown(f"**Auto-Detected in Composite ({len(comp_entry.objects)} objects):**")
                        for o in comp_entry.objects:
                            st.markdown(f"- `{o.label}` (Conf: {o.confidence:.1%}, BBox: `{o.bbox.to_dict()}`)")

                except Exception as e:
                    st.error(f"Error generating composite: {e}")


# -----------------------------------------------------------------------------
# TAB 4: Architecture & Diagnostics
# -----------------------------------------------------------------------------
elif nav == "📊 Architecture & Diagnostics":
    st.markdown('<div class="main-title">System Architecture & Technical Specifications</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed architecture, algorithms, trade-offs, and compliance verification.</div>', unsafe_allow_html=True)

    st.markdown("""
### 1. Architectural Highlights

```
┌────────────────────────────────────────────────────────────────────────┐
│                        VisionStudio Platform                           │
├────────────────────┬───────────────────────────────────────────────────┤
│ Catalog & Metadata │ • Atomic JSON storage + asset archive             │
│ Management Engine  │ • Dual-schema: AutoGenerated vs UserEditable      │
│                    │ • Invariant: Manual corrections locked as truth   │
├────────────────────┼───────────────────────────────────────────────────┤
│ Multi-Modal Visual │ • Photographic objects (saliency & watershed)     │
│ Detector           │ • Cartoon figures (anatomical clothing color map) │
│                    │ • Geometric shapes & swatches (polygons & moments)│
│                    │ • Anti-aliased RGBA alpha cutout extractor        │
├────────────────────┼───────────────────────────────────────────────────┤
│ Natural-Language   │ • Entity, quantity & spatial relationship parser  │
│ Compositor         │ • Authentic Asset Resolver (Zero hallucination)   │
│                    │ • Dynamic scaling, depth z-ordering, drop shadows │
│                    │ • Re-registers composite into catalog with prov.  │
└────────────────────┴───────────────────────────────────────────────────┘
```

### 2. Implementation of Required Scenarios

| Scenario | Requirement | Implementation Status |
|---|---|---|
| **Scenario 1** | Detect objects, colors, confidences, bboxes | Verified on `apple`, `orange`, `boy`, `girl`, `shapes`, `colors` |
| **Scenario 2** | Manually add object when detection misses | `catalog.add_manual_object()` with origin `USER_MANUAL` and entry lock |
| **Scenario 3** | Edit object name, color, confidence, bbox | `catalog.edit_object()` preserving pre-edit and auto snapshots |
| **Scenario 4** | Manual edits persist until explicit rerun | Verified: `force_rerun=False` preserves; `force_rerun=True` resets |
| **Scenario 5** | Composite with existing `apple` & `orange` | NL prompt parsed, real cutouts composited with contact shadows |
| **Scenario 6** | Composite with boy, girl, and shapes | Multi-entity layout, proportional scaling, depth z-indexing |
| **Scenario 7** | Overlapping objects & empty image | Watershed separation for overlaps; graceful `FAILED_NO_OBJECTS` for blank |
""")
