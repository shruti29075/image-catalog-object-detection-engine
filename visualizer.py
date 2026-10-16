"""
visualizer.py
-------------
Visualization and Annotation Engine.

Renders:
1. High-contrast bounding boxes with rounded header badges.
2. Distinct color-coding by provenance:
   - Green: Auto-detected
   - Blue: Manually added
   - Orange: User-edited
3. Confidence scores, color tags, and clothing attributes.
4. Composite verification side-by-side cards.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional

from models import DetectedObject, DetectionOrigin, CatalogEntry

# Provenance-specific color codes (BGR for OpenCV, RGB for PIL)
ORIGIN_COLORS_RGB = {
    DetectionOrigin.AUTO_DETECTED: (34, 197, 94),     # Vibrant Emerald Green
    DetectionOrigin.USER_MANUAL: (59, 130, 246),     # Tech Blue
    DetectionOrigin.USER_EDITED: (249, 115, 22),     # Amber Orange
}

ORIGIN_COLORS_BGR = {
    DetectionOrigin.AUTO_DETECTED: (94, 197, 34),
    DetectionOrigin.USER_MANUAL: (246, 130, 59),
    DetectionOrigin.USER_EDITED: (22, 115, 249),
}


class CatalogVisualizer:
    """Provides visual annotation and diagnostic rendering for images and detected objects."""

    @staticmethod
    def annotate_image(
        image_path: str,
        objects: List[DetectedObject],
        output_path: Optional[str] = None
    ) -> str:
        """
        Draws bounding boxes, labels, confidence scores, and origin badges on the image.
        Saves and returns the output path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Try to load a nice default font
        font = ImageFont.load_default()

        for obj in objects:
            bbox = obj.bbox
            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
            color_rgb = ORIGIN_COLORS_RGB.get(obj.origin, (34, 197, 94))

            # Bounding box outline (width 3)
            box_coords = [x, y, x + w, y + h]
            draw.rectangle(box_coords, outline=color_rgb + (255,), width=3)

            # Semitransparent fill inside bbox
            draw.rectangle(box_coords, fill=color_rgb + (25,))

            # Origin tag
            origin_tag = "[AUTO]" if obj.origin == DetectionOrigin.AUTO_DETECTED else (
                "[MANUAL]" if obj.origin == DetectionOrigin.USER_MANUAL else "[EDITED]"
            )

            # Build label text
            conf_pct = int(round(obj.confidence * 100))
            color_str = obj.color.primary_name
            text = f"{origin_tag} {obj.label} ({color_str}) {conf_pct}%"

            if obj.color.clothing_colors:
                # Add clothing info for cartoon figures
                top = obj.color.clothing_colors.get("top", "")
                bot = obj.color.clothing_colors.get("bottom", "")
                if top and bot:
                    text += f" | {top}, {bot}"

            # Calculate text size using font.getbbox
            text_bbox = font.getbbox(text)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            badge_h = text_h + 8
            badge_w = text_w + 12

            badge_y = max(0, y - badge_h - 2)
            badge_x = max(0, x)

            # Badge pill background
            draw.rectangle(
                [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                fill=color_rgb + (230,)
            )

            # Text (white)
            draw.text((badge_x + 6, badge_y + 4), text, fill=(255, 255, 255, 255), font=font)

        final_img = Image.alpha_composite(img, overlay).convert("RGB")

        out = output_path or image_path.replace(".png", "_annotated.png")
        final_img.save(out, "PNG")
        return os.path.abspath(out)

    @staticmethod
    def create_side_by_side_card(
        source_paths: List[str],
        composite_path: str,
        output_path: str,
        title: str = "Composite Image Generation Result"
    ) -> str:
        """
        Creates an executive visual comparison showing original input assets
        alongside the generated composite image.
        """
        card_w = 1200
        card_h = 750
        card = Image.new("RGB", (card_w, card_h), (248, 250, 252))
        draw = ImageDraw.Draw(card)
        font = ImageFont.load_default()

        # Header title
        draw.text((40, 25), title, fill=(15, 23, 42), font=font)

        # Left Column: Source Assets
        draw.text((40, 70), "Source Catalog Assets (Cutouts):", fill=(71, 85, 105), font=font)
        y_offset = 100
        for sp in source_paths[:3]:
            if os.path.exists(sp):
                with Image.open(sp).convert("RGBA") as src:
                    src.thumbnail((180, 180), Image.Resampling.LANCZOS)
                    # White card behind thumbnail
                    draw.rectangle([40, y_offset, 40 + 190, y_offset + 190], fill=(255, 255, 255), outline=(226, 232, 240))
                    card.paste(src, (45, y_offset + 5), src)
                    y_offset += 205

        # Right Column: Result Composite
        draw.text((270, 70), "Generated Composite Image (Saved to Catalog):", fill=(71, 85, 105), font=font)
        if os.path.exists(composite_path):
            with Image.open(composite_path).convert("RGB") as comp:
                comp.thumbnail((880, 600), Image.Resampling.LANCZOS)
                draw.rectangle([270, 100, 270 + comp.width + 10, 100 + comp.height + 10], fill=(255, 255, 255), outline=(203, 213, 225))
                card.paste(comp, (275, 105))

        card.save(output_path, "PNG")
        return os.path.abspath(output_path)
