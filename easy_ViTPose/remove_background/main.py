import math
import os

import cv2
import numpy as np

from .transparent_background import Remover


def predict_mask(remover: Remover.Remover, img):

    mask = None
    # img = Image.open(os.path.join(input_folder, input_file_name)).convert("RGB") # read image

    width, height = img.size

    scale_w = 512/width
    scale_h = 512/height
    scale = min(scale_w, scale_h)
    if scale > 1.0:
        img = img.resize((math.ceil(width*scale)-1, math.ceil(height*scale)-1))  # work around
        img = img.resize((math.ceil(width*scale), math.ceil(height*scale)))

    scale_w = 1280/width
    scale_h = 1280/height
    scale = min(scale_w, scale_h)
    if scale < 1.0:
        img = img.resize((math.ceil(width*scale), math.ceil(height*scale)))

    out = remover.process(img, threshold=0.5)  # use threhold parameter for hard prediction.
    out = out.resize((width, height))
    img = img.resize((width, height))
    out = np.array(out)[:, :, [2, 1, 0, 3]].copy()

    mask = out[:, :, 3]
    rgb = np.array(img)[:, :, [2, 1, 0]]

    return rgb, mask


def post_processing(mask) -> tuple[np.ndarray, tuple[int, int, int, int]]:

    _, binary_img = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    # edge_img = cv2.Canny(binary_img, 100, 200)

    alpha = np.zeros_like(mask)
    # kernel = np.ones((3,3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

    for iter in range(5):

        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for hid in range(len(contours)):
            if hierarchy[0][hid][3] == -1:
                # cv2.polylines(alpha, [contours[hid]], isClosed=True, color=(255, 255, 255), thickness=1)
                cv2.fillPoly(alpha, [contours[hid]], color=(255, 255, 255))
                # hull = cv2.convexHull(contours[hid])
                # cv2.fillPoly(alpha, [hull], color=(255, 255, 255))
            else:
                # cv2.polylines(alpha, [contours[hid]], isClosed=True, color=(255, 255, 255), thickness=1)
                cv2.fillPoly(alpha, [contours[hid]], color=(255, 255, 255))
                # hull = cv2.convexHull(contours[hid])
                # cv2.fillPoly(alpha, [hull], color=(255, 255, 255))

        # for contour in contours:
        #     cv2.polylines(alpha, [contour], isClosed=True, color=(255, 255, 255), thickness=1)
        #     cv2.fillPoly(alpha, [contour], color=(255, 255, 255))

        # alpha = cv2.erode(alpha, kernel)
        processed_mask = alpha

    alpha, bounding_box = get_all_components(processed_mask)

    return alpha, bounding_box # (x, y, w, h)


def get_all_components(image) -> tuple[np.ndarray[np.uint8], tuple[int, int, int, int]]:

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    # sizes = stats[:, -1]

    x_min, y_min, x_max, y_max = [output.shape[1]-1, output.shape[0]-1, 0, 0]
    obj_mask = np.zeros(output.shape, dtype=np.uint8)
    for i in range(1, nb_components):
        component = np.zeros(output.shape, dtype=np.uint8)
        component[output == i] = 255
        contours, _ = cv2.findContours(component, 1, 2)

        for hid in range(len(contours)):
            hull = cv2.convexHull(contours[hid])
            cv2.fillPoly(obj_mask, [contours[hid]], color=(255, 255, 255))

            x, y, w, h = cv2.boundingRect(hull)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x+w)
            y_max = max(y_max, y+h)

    return obj_mask, (x_min, y_min, x_max - x_min, y_max - y_min)


def get_largest_object(image) -> tuple[np.ndarray, tuple[int, int, int, int]]:

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    obj_mask = np.zeros(output.shape, dtype=np.uint8)
    obj_mask[output == max_label] = 255

    contours, _ = cv2.findContours(obj_mask, 1, 2)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    return obj_mask, (x, y, w, h)


if __name__ == "__main__":

    pass