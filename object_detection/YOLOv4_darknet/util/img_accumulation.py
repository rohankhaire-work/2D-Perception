import os
import argparse


def list_images_to_txt(folder_path, output_txt):
    # Get all image files (common formats)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    image_files = [f for f in os.listdir(
        folder_path) if f.lower().endswith(image_extensions)]

    # Sort to maintain order
    image_files.sort()

    # Write to txt file
    with open(output_txt, 'w') as f:
        for img in image_files:
            f.write(f"{os.path.join(folder_path, img)}\n")

    print(f"Saved {len(image_files)} image paths to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List all images in a folder and save to a text file.")
    parser.add_argument("folder_path", type=str,
                        help="Path to the folder containing images")
    parser.add_argument("output_txt", type=str,
                        help="Path to the output text file")

    args = parser.parse_args()

    list_images_to_txt(args.folder_path, args.output_txt)
