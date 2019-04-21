import cv2
import numpy as np


def draw_word_boxes(img, word_box_list, word_color=(255, 0, 0), letter_color=None, fill=False):
    """
    Draws boxes on the image. Inplace
    :param img: np.array - image
    :param word_box_list: list - boxes
    :param word_color: tuple or int or None - boundary color fow words
    :param letter_color: tuple or list or None - boundary color for letters
    :param fill: bool - whether to fill the boxes with color or just draw a quadrilateral
    """
    thickness = -1 if fill else 2
    for word in word_box_list:
        if word_color is not None:
            cv2.polylines(img, [word['box'][[0, 1, 3, 2], ::-1].reshape((-1, 1, 2))], True, word_color,
                          thickness=thickness)

        if letter_color is not None:
            for char in word['chars']:
                cv2.polylines(img, [char['box'][[0, 1, 3, 2], ::-1].reshape((-1, 1, 2))], True, letter_color,
                              thickness=thickness)


def get_image_from_box(image, data, height=32):
    """
    Cuts image with bounding box using perspective Transform
    :param image: numpy.ndarray: image
    :param data: dict: corresponding word data box
    :param height: int: height of the result image
    :return: (np.ndarray, list): cut image, list of char x axis delimiters
    """
    box = data['box']
    scale = np.sqrt((box[0, 1] - box[1, 1])**2 + (box[0, 0] - box[1, 0])**2) / height
    w = int(np.sqrt((box[1, 1] - box[2, 1])**2 + (box[1, 0] - box[2, 0])**2) / scale)
    pts1 = np.float32(box)[:, ::-1]
    pts1 = pts1[[1, 0, 3, 2]]
    pts2 = np.float32([[0, 0], [height, 0], [0, w],  [height, w]])[:, ::-1]
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_img = cv2.warpPerspective(image, M, (w, height))

    begin = box[0][1]
    dist = (box[2][1] - begin)
    delimiters = []
    for (char, next_char) in zip(data['chars'], data['chars'][1:]):
        left = (char['box'][3][1] - begin) / dist * w
        right = (next_char['box'][0][1] - begin) / dist * w
        delimiters.append(int((left + right) / 2))
    return result_img, delimiters


def combine_masks(true_mask, predicted_mask):
    """
    Combines true and predicted masks into one image for convenient comparison.
    :param true_mask: 2D np.ndarray - gray image with true mask
    :param predicted_mask: 2D np.ndarray - gray image with predicted mask
    :return: 3D np.ndarray - colored image with both masks
    """
    if true_mask.shape != predicted_mask.shape:
        raise ValueError("Shapes do not match")
    if true_mask.ndim != 2:
        raise ValueError("Masks should be greyscaled")

    img = np.zeros(true_mask.shape + (3,))
    img[:, :, 2] = 255 - predicted_mask
    img[:, :, 1] = 255 - true_mask
    return img
