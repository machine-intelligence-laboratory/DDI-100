import cv2
from generator import Generator
from utils import draw_word_boxes, combine_masks


def test_doc():
    gen = Generator("../data/pdf_dataset")
    img, mask, data = gen.get_doc()
    cv2.imshow("image", cv2.resize(img, (0, 0), fx=0.2, fy=0.2))
    cv2.imshow("mask", cv2.resize(mask, (0, 0), fx=0.2, fy=0.2))
    draw_word_boxes(img, data, word_color=0)
    cv2.imshow("image with boxes", cv2.resize(img, (0, 0), fx=0.2, fy=0.2))
    cv2.waitKey()


def test_str():
    gen = Generator("../data/pdf_dataset")
    img, data, delimeters = gen.get_string()
    cv2.imshow("image", cv2.resize(img, (0, 0), fx=2, fy=2))
    print(data)
    for delim in delimeters:
        cv2.line(img, (delim, 0), (delim, 32), color=0, thickness=2)
    cv2.imshow("image with delims", cv2.resize(img, (0, 0), fx=2, fy=2))
    cv2.waitKey()


def test_char():
    gen = Generator("../data/pdf_dataset")
    img, data = gen.get_char()
    cv2.imshow("image", cv2.resize(img, (0, 0), fx=2, fy=2))
    print(data)
    cv2.waitKey()


def test_mask():
    gen = Generator("../data/pdf_dataset")
    _, mask1, _ = gen.get_doc()
    _, mask2, _ = gen.get_doc()
    mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
    img = combine_masks(mask1, mask2)
    cv2.imshow("image", cv2.resize(img, (0, 0), fx=.2, fy=.2))
    cv2.waitKey()


if __name__ == "__main__":
    test_doc()
    test_str()
    test_char()
    test_mask()
