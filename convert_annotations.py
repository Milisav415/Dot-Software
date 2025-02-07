import os
import glob
from PIL import Image


def find_image_file(images_dir, img_id):
    """
    Search for an image file in images_dir that starts with img_id.
    Returns the first match found or None.
    """
    pattern = os.path.join(images_dir, f"{img_id}.*")
    files = glob.glob(pattern)
    return files[0] if files else None


def convert_annotation_file(gt_file, image_file, output_file):
    """
    Converts a single annotation file from 6 columns (x, y, w, h, class, extra)
    to YOLO format (class, x_center_norm, y_center_norm, width_norm, height_norm),
    while ignoring annotations that result in non-normalized or out-of-bounds values.

    Args:
        gt_file (str): Path to the ground-truth annotation file.
        image_file (str): Path to the corresponding image file.
        output_file (str): Path where the new annotation file will be written.
    """
    try:
        with Image.open(image_file) as img:
            image_width, image_height = img.size
    except Exception as e:
        print(f"Error opening image {image_file}: {e}")
        return

    output_lines = []
    with open(gt_file, 'r') as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"Skipping line in {gt_file} (expected 6 columns, got {len(parts)}): {line.strip()}")
                continue
            try:
                # Parse original values assuming order: x, y, width, height, class, extra
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                cls = parts[4]
                # Compute YOLO normalized values:
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                w_norm = w / image_width
                h_norm = h / image_height

                # Check that the normalized values are in valid ranges.
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w_norm <= 1 and 0 < h_norm <= 1):
                    print(f"Ignoring annotation in {gt_file} due to out-of-bounds normalization: {line.strip()}")
                    continue

                new_line = f"{0} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                output_lines.append(new_line)
            except Exception as e:
                print(f"Error processing line in {gt_file}: {line.strip()} - {e}")
                continue

    if output_lines:
        with open(output_file, 'w') as fout:
            fout.writelines(output_lines)
        print(f"Converted {gt_file} -> {output_file}")
    else:
        print(f"No valid annotations found in {gt_file}. Skipping file.")


def process_dataset_split(split_path):
    """
    Processes a dataset split (e.g., 'db/train') by converting all annotation
    files in its 'gt' folder and saving them to a new 'labels' folder.

    Args:
        split_path (str): Path to the dataset split (train, val, or test folder).
    """
    gt_dir = os.path.join(split_path, "gt")
    images_dir = os.path.join(split_path, "images")
    labels_dir = os.path.join(split_path, "labels")

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
    for gt_file in gt_files:
        base_name = os.path.splitext(os.path.basename(gt_file))[0]
        image_file = find_image_file(images_dir, base_name)
        if image_file is None:
            print(f"Image file for {base_name} not found in {images_dir}. Skipping.")
            continue
        output_file = os.path.join(labels_dir, os.path.basename(gt_file))
        convert_annotation_file(gt_file, image_file, output_file)


def process_full_dataset(root_dataset):
    """
    Processes the full dataset with splits (e.g., train, val, test) under root_dataset.

    Args:
        root_dataset (str): Root directory of your dataset (e.g., 'db').
    """
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_dataset, split)
        if os.path.isdir(split_path):
            print(f"Processing {split_path}...")
            process_dataset_split(split_path)
        else:
            print(f"Split directory {split_path} does not exist.")


if __name__ == "__main__":
    """
    after downloading jhu_crowd_v2.0 ru this script once to generate labels for training.
    """
    dataset_root = R"C:\Users\jm190\Desktop\jhu_crowd_v2.0"  # Change to your dataset root
    process_full_dataset(dataset_root)
