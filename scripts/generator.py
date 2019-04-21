import cv2
import numpy as np
import pickle

from glob import glob
from pathlib import Path
from random import choice

from .utils import get_image_from_box


class Generator:
    def __init__(self, pickle_path = None, dataset_path=None, book_paths=None):
        """
        Class Generator implements simple random sample choice.
        One of the arguments must be specified.

        :param dataset_path: str - dataset_path to full dataset
        :param book_paths: str - paths to book directories.
        """
        if dataset_path is None and book_paths is None and pickle_path is None:
            raise ValueError("One of dataset_path or book_paths should be specified")
        if dataset_path is not None:
            book_paths = glob(f"{dataset_path}/*")
        if book_paths is not None:
            self.paths = []
            for book in book_paths:
                self.paths += glob(f"{book}/gen_imgs/*")
        if pickle_path is not None:
            with open(pickle_path, "rb") as f:
                paths = pickle.load(f)
            self.paths = paths

    def get_doc(self):
        """
        Returns random document sample from dataset.

        :return: (img, masks, data)
        """
        path = choice(self.paths)
        img_path = Path(choice(self.paths))
        boxes_path = img_path.parent.parent.joinpath('gen_boxes').joinpath(img_path.stem + ".pickle")
        mask_paths = sorted(img_path.parent.parent.joinpath('gen_masks').glob(img_path.stem + '*'))
        with open(boxes_path, "rb") as f:
            data = pickle.load(f)
        img = cv2.imread(str(img_path), 0)
        masks = []
        for mask_path in mask_paths:
            masks.append(cv2.imread(str(mask_path), 0))
        return img, masks, data

    def get_string(self):
        """
        Returns sample with random single word string from dataset.

        :return: (img, str, list) - img of word, string representation, list of char x axis delimiters
        """
        img, mask, data = self.get_doc()
        word = choice(data)
        cut_img, delimiters = get_image_from_box(img, word)
        shift = np.min(word['box'], axis=0)
        for char in word['chars']:
            char['box'] -= shift
        return cut_img, word['text'], delimiters

    def get_char(self):
        """
        Returns sample with random single char string from dataset.

        :return: (img, str) - img with letter and letter
        """
        img, _, boxes = self.get_doc()
        word = choice(boxes)
        char = choice(word['chars'])
        cut_img = get_image_from_box(img, char['box'])
        return cut_img, char['text']
