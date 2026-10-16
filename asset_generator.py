"""
asset_generator.py
------------------
Programmatic Generator for the Benchmark Image Catalog Test Dataset.

Generates the 8 mandatory test images plus the empty edge-case:
1. 'apple'              - Single red apple with leaf & stem on white background
2. 'orange'             - Single orange fruit on white background
3. 'boy'                - Cartoon boy standing with blue shirt and khaki pants
4. 'girl'               - Cartoon girl standing with pink/purple outfit
5. 'shapes'             - Circle, square, triangle, and rectangle
6. 'colors'             - Red, blue, green, yellow, and purple color swatches
7. 'apple_orange'       - Red apple beside an orange fruit
8. 'overlapping_shapes' - Overlapping colored geometric shapes
9. 'empty'              - Pure white background (no detectable objects)
"""

from __future__ import annotations
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_benchmark_assets(output_dir: str = "test_assets") -> dict[str, str]:
    """Generates all 9 test assets and returns a mapping {image_name: filepath}."""
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    # 1. apple
    apple_path = os.path.join(output_dir, "apple.png")
    _create_apple_image(apple_path)
    generated["apple"] = os.path.abspath(apple_path)

    # 2. orange
    orange_path = os.path.join(output_dir, "orange.png")
    _create_orange_image(orange_path)
    generated["orange"] = os.path.abspath(orange_path)

    # 3. boy
    boy_path = os.path.join(output_dir, "boy.png")
    _create_cartoon_boy(boy_path)
    generated["boy"] = os.path.abspath(boy_path)

    # 4. girl
    girl_path = os.path.join(output_dir, "girl.png")
    _create_cartoon_girl(girl_path)
    generated["girl"] = os.path.abspath(girl_path)

    # 5. shapes
    shapes_path = os.path.join(output_dir, "shapes.png")
    _create_geometric_shapes(shapes_path)
    generated["shapes"] = os.path.abspath(shapes_path)

    # 6. colors
    colors_path = os.path.join(output_dir, "colors.png")
    _create_color_swatches(colors_path)
    generated["colors"] = os.path.abspath(colors_path)

    # 7. apple_orange
    apple_orange_path = os.path.join(output_dir, "apple_orange.png")
    _create_apple_orange_image(apple_orange_path)
    generated["apple_orange"] = os.path.abspath(apple_orange_path)

    # 8. overlapping_shapes
    overlapping_path = os.path.join(output_dir, "overlapping_shapes.png")
    _create_overlapping_shapes(overlapping_path)
    generated["overlapping_shapes"] = os.path.abspath(overlapping_path)

    # 9. empty
    empty_path = os.path.join(output_dir, "empty.png")
    _create_empty_image(empty_path)
    generated["empty"] = os.path.abspath(empty_path)

    return generated


def _create_apple_image(path: str):
    """Draws a crisp red apple on white background."""
    w, h = 600, 600
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    # Stem
    draw.line([(300, 150), (290, 110)], fill=(101, 67, 33, 255), width=8)

    # Leaf
    leaf_poly = [(290, 125), (345, 105), (370, 125), (330, 140)]
    draw.polygon(leaf_poly, fill=(34, 197, 94, 255))

    # Apple Body (Red with slight highlight)
    # Dual lobe shape
    draw.ellipse([180, 160, 360, 440], fill=(220, 38, 38, 255))
    draw.ellipse([240, 160, 420, 440], fill=(220, 38, 38, 255))
    draw.ellipse([200, 200, 400, 470], fill=(210, 25, 25, 255))

    # Subtle specular highlight (light red/white sheen)
    draw.ellipse([220, 190, 270, 270], fill=(248, 113, 113, 255))

    im.convert("RGB").save(path, "PNG")


def _create_orange_image(path: str):
    """Draws a vibrant orange fruit on white background."""
    w, h = 600, 600
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    # Small green stem calyx
    draw.ellipse([290, 135, 310, 155], fill=(22, 163, 74, 255))

    # Main orange fruit body
    draw.ellipse([175, 150, 425, 450], fill=(234, 88, 12, 255))

    # Inner warm tone
    draw.ellipse([195, 170, 405, 430], fill=(249, 115, 22, 255))

    # Highlight sheen
    draw.ellipse([215, 185, 275, 245], fill=(251, 146, 60, 255))

    im.convert("RGB").save(path, "PNG")


def _create_cartoon_boy(path: str):
    """Draws a full-body standing cartoon boy with blue shirt and khaki pants."""
    w, h = 500, 700
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    # Head & Face
    draw.ellipse([205, 90, 295, 185], fill=(245, 210, 180, 255))  # Skin tone
    # Hair (Brown)
    draw.ellipse([200, 80, 300, 135], fill=(90, 50, 20, 255))
    # Eyes & Smile
    draw.ellipse([225, 130, 235, 142], fill=(20, 20, 20, 255))
    draw.ellipse([265, 130, 275, 142], fill=(20, 20, 20, 255))
    draw.arc([235, 145, 265, 165], 0, 180, fill=(180, 50, 50, 255), width=3)

    # Torso: Bright Blue Shirt
    draw.rectangle([185, 185, 315, 360], fill=(37, 99, 235, 255))
    # Sleeves/Arms
    draw.rectangle([150, 195, 185, 320], fill=(37, 99, 235, 255))
    draw.rectangle([315, 195, 350, 320], fill=(37, 99, 235, 255))
    # Hands (skin)
    draw.ellipse([145, 320, 180, 345], fill=(245, 210, 180, 255))
    draw.ellipse([320, 320, 355, 345], fill=(245, 210, 180, 255))

    # Lower Garment: Khaki Pants
    draw.rectangle([195, 360, 305, 550], fill=(180, 150, 100, 255))
    # Leg division line
    draw.line([(250, 420), (250, 550)], fill=(255, 255, 255, 255), width=4)

    # Shoes (Dark Brown)
    draw.ellipse([185, 545, 245, 580], fill=(60, 40, 20, 255))
    draw.ellipse([255, 545, 315, 580], fill=(60, 40, 20, 255))

    im.convert("RGB").save(path, "PNG")


def _create_cartoon_girl(path: str):
    """Draws a full-body standing cartoon girl with pink/purple dress."""
    w, h = 500, 700
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    # Long Dark Hair (back)
    draw.ellipse([180, 80, 320, 280], fill=(40, 25, 20, 255))

    # Head & Face
    draw.ellipse([205, 95, 295, 190], fill=(245, 210, 180, 255))
    # Front bangs
    draw.ellipse([200, 85, 300, 125], fill=(40, 25, 20, 255))
    # Eyes & Smile
    draw.ellipse([225, 135, 235, 147], fill=(20, 20, 20, 255))
    draw.ellipse([265, 135, 275, 147], fill=(20, 20, 20, 255))
    draw.arc([235, 152, 265, 170], 0, 180, fill=(219, 39, 119, 255), width=3)

    # Torso: Pink Top
    draw.rectangle([195, 190, 305, 300], fill=(236, 72, 153, 255))
    # Arms
    draw.rectangle([160, 200, 195, 310], fill=(245, 210, 180, 255))
    draw.rectangle([305, 200, 340, 310], fill=(245, 210, 180, 255))

    # Lower Garment: Purple Skirt (Trapezoid)
    skirt_poly = [(195, 300), (305, 300), (335, 430), (165, 430)]
    draw.polygon(skirt_poly, fill=(147, 51, 234, 255))

    # Legs (skin)
    draw.rectangle([210, 430, 235, 545], fill=(245, 210, 180, 255))
    draw.rectangle([265, 430, 290, 545], fill=(245, 210, 180, 255))

    # Shoes (Violet/Pink)
    draw.ellipse([200, 540, 245, 575], fill=(124, 58, 237, 255))
    draw.ellipse([255, 540, 300, 575], fill=(124, 58, 237, 255))

    im.convert("RGB").save(path, "PNG")


def _create_geometric_shapes(path: str):
    """Draws 4 distinct geometric shapes: Circle, Square, Triangle, Rectangle."""
    w, h = 900, 450
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    # 1. Circle (Blue)
    draw.ellipse([70, 130, 230, 290], fill=(37, 99, 235, 255))

    # 2. Square (Green)
    draw.rectangle([270, 130, 430, 290], fill=(22, 163, 74, 255))

    # 3. Triangle (Red)
    tri_poly = [(550, 120), (470, 290), (630, 290)]
    draw.polygon(tri_poly, fill=(220, 38, 38, 255))

    # 4. Rectangle (Yellow / Gold)
    draw.rectangle([670, 160, 850, 260], fill=(234, 179, 8, 255))

    im.convert("RGB").save(path, "PNG")


def _create_color_swatches(path: str):
    """Draws five distinct color swatches: Red, Blue, Green, Yellow, Purple."""
    w, h = 1000, 380
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    colors = [
        ("red", (220, 38, 38)),
        ("blue", (37, 99, 235)),
        ("green", (22, 163, 74)),
        ("yellow", (234, 179, 8)),
        ("purple", (147, 51, 234))
    ]

    swatch_w = 140
    swatch_h = 220
    gap = 45
    start_x = 55
    y1 = 80
    y2 = y1 + swatch_h

    for i, (name, rgb) in enumerate(colors):
        x1 = start_x + i * (swatch_w + gap)
        x2 = x1 + swatch_w
        # Rounded swatch card
        draw.rectangle([x1, y1, x2, y2], fill=rgb + (255,))

    im.convert("RGB").save(path, "PNG")


def _create_apple_orange_image(path: str):
    """Draws an apple beside an orange fruit on a white surface."""
    w, h = 800, 500
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    # Apple on Left (x: 140 to 360)
    # Stem
    draw.line([(250, 160), (242, 120)], fill=(101, 67, 33, 255), width=7)
    # Leaf
    draw.polygon([(242, 135), (285, 120), (305, 135), (275, 150)], fill=(34, 197, 94, 255))
    # Apple Body
    draw.ellipse([150, 170, 290, 390], fill=(220, 38, 38, 255))
    draw.ellipse([210, 170, 350, 390], fill=(220, 38, 38, 255))
    draw.ellipse([170, 200, 330, 410], fill=(200, 25, 25, 255))

    # Orange on Right (x: 440 to 660)
    # Calyx
    draw.ellipse([540, 165, 560, 185], fill=(22, 163, 74, 255))
    # Body
    draw.ellipse([440, 175, 660, 400], fill=(234, 88, 12, 255))
    draw.ellipse([460, 195, 640, 385], fill=(249, 115, 22, 255))

    im.convert("RGB").save(path, "PNG")


def _create_overlapping_shapes(path: str):
    """Draws multiple overlapping colored geometric shapes."""
    w, h = 750, 500
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    # 1. Base Layer: Cyan Circle
    draw.ellipse([150, 140, 370, 360], fill=(14, 165, 233, 255))

    # 2. Overlapping Magenta / Purple Square
    draw.rectangle([280, 180, 480, 380], fill=(147, 51, 234, 255))

    # 3. Overlapping Amber / Yellow Triangle on top
    tri_poly = [(500, 100), (390, 320), (610, 320)]
    draw.polygon(tri_poly, fill=(234, 179, 8, 255))

    im.convert("RGB").save(path, "PNG")


def _create_empty_image(path: str):
    """Draws a pure white empty image with zero detectable objects."""
    w, h = 600, 600
    im = Image.new("RGB", (w, h), (255, 255, 255))
    im.save(path, "PNG")


if __name__ == "__main__":
    assets = generate_benchmark_assets()
    print("Benchmark assets generated successfully:")
    for name, p in assets.items():
        print(f"  - {name}: {p}")
