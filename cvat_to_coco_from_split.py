import os
import json
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

def load_split_coco(split_path):
    """Load an existing COCO split and build quick lookup tables."""
    with open(split_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    img_by_name = {img["file_name"]: img for img in images}

    ann_by_image = {}
    for ann in annotations:
        ann_by_image[ann["image_id"]] = ann

    return coco, img_by_name, ann_by_image


def parse_body_polygon(poly_str):
    coords = []
    for pair in poly_str.strip().split(";"):
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        coords.append(float(x_str))
        coords.append(float(y_str))
    return coords


def polygon_to_bbox_and_area(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    pts = list(zip(xs, ys))
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0

    return bbox, area


def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    task = root.find("meta").find("task")
    task_name = task.find("name").text

    result = {}

    for img in root.findall("image"):
        img_name = img.attrib["name"]
        width = int(img.attrib["width"])
        height = int(img.attrib["height"])

        tag_el = img.find("tag")
        tag_label = tag_el.attrib["label"] if tag_el is not None else None

        body_poly_coords = None
        for poly_el in img.findall("polygon"):
            if poly_el.attrib.get("label") == "Body":
                body_poly_coords = parse_body_polygon(poly_el.attrib["points"])
                break

        kp_dict = {}
        for p in img.findall("points"):
            label = p.attrib.get("label")
            if not label:
                continue
            try:
                idx = int(label)
            except ValueError:
                continue
            x_str, y_str = p.attrib["points"].split(",")
            kp_dict[idx] = (float(x_str), float(y_str))

        result[(task_name, img_name)] = {
            "width": width,
            "height": height,
            "tag": tag_label,
            "polygon": body_poly_coords,
            "kp_dict": kp_dict,
        }

    return result


def load_all_cvat(xml_root):
    all_images = {}
    for fname in os.listdir(xml_root):
        if not fname.lower().endswith(".xml"):
            continue
        path = os.path.join(xml_root, fname)
        part = parse_cvat_xml(path)
        all_images.update(part)
    return all_images


def build_coco_from_cvat(xml_root, split_json, output_json):
    coco, img_by_name, ann_by_image = load_split_coco(split_json)
    cvat_index = load_all_cvat(xml_root)

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": [],
        "annotations": [],
    }

    missing = []

    for img in coco["images"]:
        file_name = img["file_name"]
        img_id = img["id"]
        old_ann = ann_by_image.get(img_id)

        # -------------------------------
        #  FIX: Special-case detection
        # -------------------------------
        # If the COCO path has: "2022 2023 General/images/FILE"
        # then our XML key will be ("2022 2023 General", FILE)
        if file_name.startswith("2022 2023 General/images/"):
            task_name = "2022 2023 General"
            image_file = file_name.split("/")[-1]
            key = (task_name, image_file)
        else:
            # normal behavior:
            # file_name = "2018 Peces Ituango/IMG_0001.jpg"
            parts = file_name.split("/")
            task_name = parts[0]
            image_file = parts[-1]
            key = (task_name, image_file)

        if key not in cvat_index:
            missing.append(file_name)
            continue

        cvat_info = cvat_index[key]

        # ---- IMAGE ----
        new_img = dict(img)
        if cvat_info["tag"] is not None:
            new_img["label"] = cvat_info["tag"]
        new_coco["images"].append(new_img)

        # ---- KEYPOINTS ----
        kp_list = []
        num_kp = 0
        for k in range(1, 21):
            if k in cvat_info["kp_dict"]:
                x, y = cvat_info["kp_dict"][k]
                v = 2
                num_kp += 1
            else:
                x = y = 0.0
                v = 0
            kp_list.extend([x, y, v])

        # ---- BBOX ----
        if cvat_info["polygon"] is not None:
            bbox, area = polygon_to_bbox_and_area(cvat_info["polygon"])
        else:
            bbox = old_ann.get("bbox", [0, 0, 0, 0]) if old_ann else [0, 0, 0, 0]
            area = old_ann.get("area", 0.0) if old_ann else 0.0

        # ---- ANNOTATION ----
        ann_id = old_ann["id"] if old_ann else len(new_coco["annotations"]) + 1
        cat_id = old_ann["category_id"] if old_ann else 1

        new_ann = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "keypoints": kp_list,
            "num_keypoints": num_kp,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        }
        new_coco["annotations"].append(new_ann)

    # ---- SAVE ----
    with open(output_json, "w") as f:
        json.dump(new_coco, f, indent=2)

    print(f"Wrote COCO to: {output_json}")
    print(f"Images in split: {len(coco['images'])}")
    print(f"Images with CVAT annotations: {len(new_coco['images'])}")
    if missing:
        print(f"WARNING: {len(missing)} images not found in XML.")
        print("Example:", missing[:10])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_root", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    build_coco_from_cvat(args.xml_root, args.split_json, args.output_json)


if __name__ == "__main__":
    main()
