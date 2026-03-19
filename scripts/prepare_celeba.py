"""Convert Kaggle CelebA CSV files to torchvision-expected TXT format.

Run once after extracting the Kaggle archive:
    env/bin/python scripts/prepare_celeba.py

Expected input structure (after unzip):
    data/celeba/img_align_celeba/img_align_celeba/*.jpg  (double-nested from Kaggle)
    data/celeba/list_eval_partition.csv
    data/celeba/list_attr_celeba.csv
    data/celeba/list_bbox_celeba.csv
    data/celeba/list_landmarks_align_celeba.csv

Output structure (torchvision-compatible):
    data/celeba/img_align_celeba/*.jpg  (flattened)
    data/celeba/list_eval_partition.txt
    data/celeba/identity_CelebA.txt
    data/celeba/list_attr_celeba.txt
    data/celeba/list_bbox_celeba.txt
    data/celeba/list_landmarks_align_celeba.txt
"""

import csv
import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "celeba")


def flatten_images():
    """Fix double-nested img_align_celeba directory from Kaggle zip."""
    nested = os.path.join(DATA_DIR, "img_align_celeba", "img_align_celeba")
    target = os.path.join(DATA_DIR, "img_align_celeba")

    if os.path.isdir(nested):
        print("Flattening double-nested image directory...")
        for fname in os.listdir(nested):
            src = os.path.join(nested, fname)
            dst = os.path.join(target, fname)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        os.rmdir(nested)
        print(f"Done. Images in {target}")
    else:
        print("Images already flattened.")

    count = len([f for f in os.listdir(target) if f.endswith(".jpg")])
    print(f"Total images: {count}")


def convert_partition_csv():
    """Convert list_eval_partition.csv to .txt format.

    CSV: image_id,partition
    TXT: image_id partition
    """
    csv_path = os.path.join(DATA_DIR, "list_eval_partition.csv")
    txt_path = os.path.join(DATA_DIR, "list_eval_partition.txt")

    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} not found")
        return

    with open(csv_path) as f_in, open(txt_path, "w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            f_out.write(f"{row['image_id']} {row['partition']}\n")
    print(f"Created {txt_path}")


def convert_attr_csv():
    """Convert list_attr_celeba.csv to .txt format.

    CSV: image_id,attr1,attr2,...
    TXT: (header lines) then image_id  attr1 attr2 ...
    """
    csv_path = os.path.join(DATA_DIR, "list_attr_celeba.csv")
    txt_path = os.path.join(DATA_DIR, "list_attr_celeba.txt")

    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} not found")
        return

    with open(csv_path) as f_in:
        reader = csv.reader(f_in)
        header = next(reader)
        attr_names = header[1:]  # skip image_id
        rows = list(reader)

    with open(txt_path, "w") as f_out:
        f_out.write(f"{len(rows)}\n")
        f_out.write("  ".join(attr_names) + "\n")
        for row in rows:
            f_out.write(f"{row[0]}  {'  '.join(row[1:])}\n")
    print(f"Created {txt_path}")


def convert_bbox_csv():
    """Convert list_bbox_celeba.csv to .txt format."""
    csv_path = os.path.join(DATA_DIR, "list_bbox_celeba.csv")
    txt_path = os.path.join(DATA_DIR, "list_bbox_celeba.txt")

    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} not found")
        return

    with open(csv_path) as f_in:
        reader = csv.reader(f_in)
        header = next(reader)
        col_names = header[1:]
        rows = list(reader)

    with open(txt_path, "w") as f_out:
        f_out.write(f"{len(rows)}\n")
        f_out.write("  ".join(col_names) + "\n")
        for row in rows:
            f_out.write(f"{row[0]}  {'  '.join(row[1:])}\n")
    print(f"Created {txt_path}")


def convert_landmarks_csv():
    """Convert list_landmarks_align_celeba.csv to .txt format."""
    csv_path = os.path.join(DATA_DIR, "list_landmarks_align_celeba.csv")
    txt_path = os.path.join(DATA_DIR, "list_landmarks_align_celeba.txt")

    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} not found")
        return

    with open(csv_path) as f_in:
        reader = csv.reader(f_in)
        header = next(reader)
        col_names = header[1:]
        rows = list(reader)

    with open(txt_path, "w") as f_out:
        f_out.write(f"{len(rows)}\n")
        f_out.write("  ".join(col_names) + "\n")
        for row in rows:
            f_out.write(f"{row[0]}  {'  '.join(row[1:])}\n")
    print(f"Created {txt_path}")


def create_identity_file():
    """Create identity_CelebA.txt (torchvision requires it but we don't use it).

    If not available, create a dummy with all images assigned identity 1.
    """
    txt_path = os.path.join(DATA_DIR, "identity_CelebA.txt")
    if os.path.exists(txt_path):
        print(f"Already exists: {txt_path}")
        return

    img_dir = os.path.join(DATA_DIR, "img_align_celeba")
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    with open(txt_path, "w") as f_out:
        for img in images:
            f_out.write(f"{img} 1\n")
    print(f"Created {txt_path} (dummy identities for {len(images)} images)")


def main():
    print(f"Data directory: {DATA_DIR}\n")

    flatten_images()
    print()

    convert_partition_csv()
    convert_attr_csv()
    convert_bbox_csv()
    convert_landmarks_csv()
    create_identity_file()

    print("\nDone! Dataset ready for torchvision.datasets.CelebA")


if __name__ == "__main__":
    main()
