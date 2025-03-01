import os
import json
import argparse
from glob import glob


def yolo_to_coco(image_folder, label_folder, output_json):
    # Define class names (modify this for your dataset)
    classes = ["car", "person", "truck"]

    # Get all label files
    label_files = glob(os.path.join(label_folder, "*.txt"))

    # Initialize COCO structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
    }

    annotation_id = 1  # Unique annotation ID

    # Iterate through label files
    for image_id, label_file in enumerate(label_files, start=1):
        base_name = os.path.splitext(os.path.basename(label_file))[
            0]  # Get filename without extension

        # Check for both JPG and PNG versions
        image_path = None
        for ext in [".jpg", ".png"]:
            temp_path = os.path.join(image_folder, base_name + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if image_path is None:
            print(f"⚠️ Warning: No image found for {label_file}. Skipping.")
            continue

        # Taken from metadata
        height, width = 800, 600

        # Add image info to COCO JSON
        coco_data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        })

        # Read label file
        with open(label_file, "r") as f:
            lines = f.readlines()

        # Process each bounding box
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            # Convert YOLO format (normalized) to COCO format (absolute)
            x_min = int((x_center - box_w / 2) * width)
            y_min = int((y_center - box_h / 2) * height)
            box_w = int(box_w * width)
            box_h = int(box_h * height)

            # Add annotation to COCO JSON
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x_min, y_min, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0
            })
            annotation_id += 1

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"✅ Conversion complete! COCO annotations saved to {output_json}")


# Argument parser setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YOLO TXT annotations to COCO format.")
    parser.add_argument("--image_folder", required=True,
                        help="Path to folder containing images (JPG/PNG).")
    parser.add_argument("--label_folder", required=True,
                        help="Path to folder containing YOLO TXT labels.")
    parser.add_argument("--output_json", required=True,
                        help="Output COCO JSON file path.")

    args = parser.parse_args()

    # Run the conversion
    yolo_to_coco(args.image_folder, args.label_folder, args.output_json)

