import cv2
import random
import pickle

from glob import glob
from pathlib import Path
from utils import draw_word_boxes
from generator import Generator


def random_show(gen):
    img, mask, data = gen.get_doc()
    mask = mask.copy()
    cv2.imshow("image", cv2.resize(img, (0, 0), fx=0.2, fy=0.2))
    cv2.imshow("mask", cv2.resize(mask, (0, 0), fx=0.2, fy=0.2))

    draw_word_boxes(img, data, word_color=0)
    cv2.imshow("image with boxes", cv2.resize(img, (0, 0), fx=0.2, fy=0.2))
    cv2.waitKey(1000)


def show_dataset(dataset_path):
    gen = Generator(dataset_path)
    while True:
        random_show(gen)


if __name__ == "__main__":
    dataset_path = "../data/pdf_dataset"
    show_dataset(dataset_path)
