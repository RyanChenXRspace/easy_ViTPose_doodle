import json
import re
import cv2
import numpy as np
import os
import urllib.parse
import shutil


def main(label_groups: list[str]):

    dataset_name = 'train'

    labels = merge_label_json(label_groups)
    json_output_path, total_num_data = prepare_image_dataset(labels, dataset_name)

    plot_img_folder = os.path.abspath(f"{dataset_name}_plot")
    os.makedirs(plot_img_folder, exist_ok=True)

    with open(f'{json_output_path}', 'r', encoding='utf-8') as f:
        res = json.load(f)
        print(f'total {len(res)} data')
        print(f'keys: {res[0].keys()}')

    for data in res:
        # data = res[0]
        normalized_keypoints = data.get('lmk')
        file_name = data.get('file')

        if normalized_keypoints:
            img = cv2.imread(
                os.path.join(f'./{dataset_name}', file_name),
                cv2.IMREAD_COLOR
            )
            img = plot_joint_on_img(img, normalized_keypoints)
            cv2.imwrite(
                os.path.join(plot_img_folder, f"{os.path.splitext(file_name)[0]}_disp.png"),
                img,
            )

    return


def resize_image(image_path, max_size=1024):

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}.")
        return None

    height, width = img.shape[:2]

    max_dimension = max(height, width)
    if max_dimension <= max_size:
        return img

    ratio = max_size / max_dimension

    new_height = int(height * ratio)
    new_width = int(width * ratio)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_img


def prepare_image_dataset(labels, out_folder):

    out_img_folder = os.path.abspath(f"./{out_folder}")
    os.makedirs(out_img_folder, exist_ok=True)

    label_data_list = []
    id_set = set()

    for data in labels:
        img_id = data.get('id')
        if img_id is None:
            print("Error: Missing 'id' in data.")
            continue

        if img_id in id_set:
            print(f"Warning: Duplicate ID found: {img_id}")
            continue

        id_set.add(img_id)

        normalized_keypoints = extract_normalized_keypoints(data)

        ori_img_path = parse_image_path(data)
        if not ori_img_path:
            print(f"Error: Unable to parse image path for ID {img_id}.")
            continue

        ori_img_ext = os.path.splitext(ori_img_path)[1]
        new_img_name = f'{img_id}{ori_img_ext}'

        label_data = {
            'file': new_img_name,
            'orifile': ori_img_path,
            'lmk': normalized_keypoints
        }

        try:
            destination_path = os.path.join(out_img_folder, new_img_name)
            shutil.copy(ori_img_path, destination_path)

            img = resize_image(destination_path)
            cv2.imwrite(destination_path, img)

            print(f"File copied and renamed to: {destination_path}")
            label_data_list.append(label_data)

        except Exception as e:
            print(f"An error occurred: {e}")

    json_output_path = os.path.join(out_img_folder, f'{out_folder}.json')
    try:
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(label_data_list, json_file, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error writing JSON file: {e}")

    return json_output_path, len(label_data_list)


def parse_image_path(data: dict) -> str:
    """
    Parse the image path from the format "/data/local-files/?d=data/img3_1.png"
    to "./data/img3_1.png"
    """

    if not isinstance(data, dict) or 'img' not in data:
        raise ValueError("Input must be a dictionary with an 'img' key")

    img_path = data.get('img')

    match = re.search(r'\?d=(.*)', img_path)
    if match:
        # print(f'{match}')
        decoded_string = urllib.parse.unquote(match.group(1))
        print(f'{decoded_string}')
        return f"./{decoded_string}"

    print("Error: The image path format is incorrect.")
    return None


def plot_joint_on_img(img: np.ndarray, keypoints: dict) -> np.ndarray:

    if img is None:
        print("Error: Invalid image input.")
        return None

    if not isinstance(keypoints, dict):
        print("Error: Keypoints must be a dictionary.")
        return img

    height, width = img.shape[:2]

    for label, coords in keypoints.items():
        if len(coords) != 2:
            print(f"Warning: Invalid coordinates for {label}. Expected (x, y), got {coords}")
            continue

        x_coord = min(max(round(coords[0] * width), 0), width)
        y_coord = min(max(round(coords[1] * height), 0), height)

        cv2.circle(img, (x_coord, y_coord), 3, (0, 0, 255), -1)

    return img


def load_img(data: dict) -> np.ndarray:

    img_path = parse_image_path(data)
    print(f'Load: {img_path}')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    return img


def extract_normalized_keypoints(data: dict):

    keypoints = {}

    result = data.get('kp-1', [])
    if not result:
        print("No annotations found in the data.")
        return keypoints

    for value in result:
        keypointlabels = value.get('keypointlabels', [])

        if keypointlabels:
            label = keypointlabels[0]
            x = value.get('x')
            y = value.get('y')
            if x is not None and y is not None:
                # Correct normalization: divide x by width and y by height
                keypoints[label] = (x / 100, y / 100)
            else:
                print(f"Warning: Missing x or y coordinate for {label}")

    return keypoints


def merge_label_json(label_groups: list[str]):

    total_labels = []

    for label_path in label_groups:
        with open(label_path, encoding='utf-8') as f:
            res = json.load(f)
            print(f'Add {len(res)} labels')
            # print(f'keys: {res[0].keys()}')

            total_labels += res

    return total_labels


if __name__ == '__main__':

    # JSON-MIN format from label studio v1.13.1
    export_list = [
        './label_json/project-1-at-2024-09-19-09-31-b014653a.json',
        './label_json/project-7-at-2024-09-19-09-17-b447e130.json'
    ]
    main(export_list)
