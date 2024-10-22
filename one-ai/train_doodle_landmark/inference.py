import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.inference import NumpyEncoder
from easy_ViTPose.vit_utils.visualization import joints_dict
from PIL import Image

try:
    import onnxruntime  # noqa: F401
    has_onnx = True
except ModuleNotFoundError:
    has_onnx = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='path to image / video or webcam ID (=cv2)')
    parser.add_argument('--output-path', type=str, default='',
                        help='output path, if the path provided is a directory '
                        'output files are "input_name +_result{extension}".')
    parser.add_argument('--model', type=str, required=True,
                        help='checkpoint path of the model')
    parser.add_argument('--yolo', type=str, required=False, default=None,
                        help='checkpoint path of the yolo model')
    parser.add_argument('--dataset', type=str, required=False, default=None,
                        help='Name of the dataset. If None it"s extracted from the file name. \
                              ["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]')
    parser.add_argument('--det-class', type=str, required=False, default=None,
                        help='["human", "cat", "dog", "horse", "sheep", \
                               "cow", "elephant", "bear", "zebra", "giraffe", "animals"]')
    parser.add_argument('--model-name', type=str, required=False, choices=['s', 'b', 'l', 'h'],
                        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument('--yolo-size', type=int, required=False, default=320,
                        help='YOLOv8 image size during inference')
    parser.add_argument('--conf-threshold', type=float, required=False, default=0.5,
                        help='Minimum confidence for keypoints to be drawn. [0, 1] range')
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270],
                        required=False, default=0,
                        help='Rotate the image of [90, 180, 270] degress counterclockwise')
    parser.add_argument('--yolo-step', type=int,
                        required=False, default=1,
                        help='The tracker can be used to predict the bboxes instead of yolo for performance, '
                             'this flag specifies how often yolo is applied (e.g. 1 applies yolo every frame). '
                             'This does not have any effect when is_video is False')
    parser.add_argument('--single-pose', default=False, action='store_true',
                        help='Do not use SORT tracker because single pose is expected in the video')
    parser.add_argument('--show', default=False, action='store_true',
                        help='preview result during inference')
    parser.add_argument('--show-yolo', default=False, action='store_true',
                        help='draw yolo results')
    parser.add_argument('--show-raw-yolo', default=False, action='store_true',
                        help='draw yolo result before that SORT is applied for tracking'
                        ' (only valid during video inference)')
    parser.add_argument('--save-img', default=False, action='store_true',
                        help='save image results')
    parser.add_argument('--save-json', default=False, action='store_true',
                        help='save json results')
    args = parser.parse_args()

    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()

    # Load Yolo
    yolo = args.yolo
    if yolo is None:
        yolo = 'easy_ViTPose/' + ('yolov8s' + ('.onnx' if has_onnx and not (use_mps or use_cuda) else '.pt'))
    input_path = args.input
    ext = input_path[input_path.rfind('.'):]

    assert not (args.save_img or args.save_json) or args.output_path, \
        'Specify an output path if using save-img or save-json flags'
    output_path = args.output_path
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    reader = []
    image_name = []
    if os.path.isfile(input_path):
        reader = [np.array(Image.open(input_path).rotate(args.rotate))[..., :3]]  # type: ignore
        image_name = [Path(input_path).name]
    elif os.path.isdir(input_path):
        folder_path = Path(input_path)
        image_files = [file for file in folder_path.iterdir() if file.suffix in ['.png', '.jpg']]

        reader = []
        image_name = []
        for image_file in image_files:
            image = np.array(Image.open(image_file).rotate(args.rotate))[..., :3]
            if image.size:
                reader.append(image)
                image_name.append(image_file.name)

    # Initialize model
    model = VitInference(args.model, yolo, args.model_name,
                         args.det_class, args.dataset,
                         args.yolo_size, is_video=False,
                         single_pose=args.single_pose,
                         yolo_step=args.yolo_step)  # type: ignore
    print(f">>> Model loaded: {args.model}")

    print(f'>>> Running inference on {input_path}')
    keypoints = []

    for ith, (img, img_name) in tqdm.tqdm(
        enumerate(zip(reader, image_name)), total=len(reader)
    ):

        img_keypoints = model.inference(img)

        if args.save_img:
            # Draw result and transform to BGR
            img = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]

            print('>>> Saving output image')
            output_img_path = os.path.join(output_path, f'{Path(img_name).stem}_result.png')
            cv2.imwrite(output_img_path, img)

        if args.save_json:
            print('>>> Saving output json')
            output_json_path = os.path.join(output_path, f'{Path(img_name).stem}_result.json')
            with open(output_json_path, 'w') as f:
                out = {
                    "keypoints": img_keypoints,
                    "skeleton": joints_dict()[model.dataset]["keypoints"],
                }
                json.dump(out, f, cls=NumpyEncoder)
