"""
Generate placeholder food test images using PIL.
Creates distinct, categorized placeholder images with food labels
for pipeline evaluation without requiring large external binary downloads.
"""
import csv
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Color palette mapped by food category
CATEGORY_COLORS = {
    "fast_food": ((220, 80, 50), (255, 220, 180)),
    "dessert": ((215, 120, 150), (255, 240, 245)),
    "seafood": ((70, 130, 180), (230, 245, 255)),
    "salad": ((60, 160, 80), (235, 255, 235)),
    "noodles": ((200, 140, 60), (255, 245, 220)),
    "meat": ((160, 50, 40), (255, 230, 225)),
    "mexican": ((210, 110, 40), (255, 240, 210)),
    "bakery": ((195, 135, 75), (255, 245, 230)),
    "rice": ((210, 160, 90), (255, 250, 240)),
    "pasta": ((225, 175, 75), (255, 250, 225)),
    "breakfast": ((230, 185, 40), (255, 250, 210)),
    "curry": ((190, 100, 25), (255, 235, 215)),
    "soup": ((180, 90, 60), (255, 235, 230)),
    "dip": ((185, 150, 90), (250, 245, 230)),
}
DEFAULT_COLORS = ((128, 128, 128), (240, 240, 240))


def generate_food_image(food_name: str, category: str, output_path: Path, size=(380, 380)):
    """Generate a single food placeholder image with text and decorative plate."""
    bg_color, plate_color = CATEGORY_COLORS.get(category, DEFAULT_COLORS)

    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw centered circular plate
    center_x, center_y = size[0] // 2, size[1] // 2
    plate_radius = int(size[0] * 0.42)
    inner_radius = int(plate_radius * 0.88)

    draw.ellipse(
        [center_x - plate_radius, center_y - plate_radius, center_x + plate_radius, center_y + plate_radius],
        fill=plate_color,
        outline=(255, 255, 255),
        width=4,
    )

    draw.ellipse(
        [center_x - inner_radius, center_y - inner_radius, center_x + inner_radius, center_y + inner_radius],
        fill=bg_color,
        outline=(200, 200, 200),
        width=2,
    )

    # Label text
    display_title = food_name.replace("_", " ").title()
    category_label = f"[{category.upper()}]"

    # Use default bitmap font with positioning
    font = ImageFont.load_default()

    # Draw banner badge
    banner_w, banner_h = int(size[0] * 0.75), 50
    banner_x1 = (size[0] - banner_w) // 2
    banner_y1 = center_y - 25
    draw.rounded_rectangle(
        [banner_x1, banner_y1, banner_x1 + banner_w, banner_y1 + banner_h],
        radius=8,
        fill=(20, 25, 35),
        outline=(255, 255, 255),
        width=2,
    )

    # Draw text
    draw.text((center_x, center_y - 8), display_title, fill=(255, 255, 255), font=font, anchor="mm")
    draw.text((center_x, center_y + 12), category_label, fill=(180, 220, 255), font=font, anchor="mm")

    # Corner label
    draw.text((12, 12), "Multimodal Food Agent Test Sample", fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="JPEG", quality=95)
    return output_path


def generate_all_samples():
    """Reads expected_labels.csv and generates all 20 sample images."""
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir.parent / "expected_labels.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected labels CSV not found at {csv_path}")

    created = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            food_name = row["food_name"]
            category = row["category"]
            out_file = base_dir / f"{food_name}.jpg"
            generate_food_image(food_name, category, out_file)
            created.append(out_file)

    print(f"Successfully generated {len(created)} test images in {base_dir}")
    return created


if __name__ == "__main__":
    generate_all_samples()
