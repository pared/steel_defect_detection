import pandas

import cv2, numpy as np
import os

from src.utils import PROJECT_ROOT


class Data(object):
    def __init__(self):
        data_path = PROJECT_ROOT / "data" / "interim"
        self.train_images_path = data_path / "train_images"

        annotations_path = data_path / "train.csv"
        self.annotations = pandas.read_csv(annotations_path)

        self.image_names = os.listdir(self.train_images_path)

    def load_image(self, image_name):
        image_path = self.train_images_path / image_name
        img = cv2.imread(str(image_path))
        return img

    def get_image_annotations(self, image_name):
        """Returns map defect_id -> encoding for image name"""

        image_rows = self.annotations[
            self.annotations['ImageId_ClassId'].str.startswith(image_name)]

        # NOTE there are 4 defects types, there MUST be 4 rows
        assert len(image_rows) == 4

        result = {}
        for name, defect_encoding in image_rows.values:
            key = int(name.split('_')[1])
            if defect_encoding is np.nan:
                defect_encoding = None
            else:
                defect_encoding = [int(e) for e in defect_encoding.split(" ")]
            result[key] = defect_encoding
        return result

    def get_defects_masks(self, image_name):
        image = self.load_image(image_name)
        annotations = self.get_image_annotations(image_name)

        result = {}

        im_h, im_w, channels = image.shape
        for defect_id, encoded_mask in annotations.items():
            if encoded_mask:
                bitmap_flat = np.zeros(im_w * im_h)
                for i in range(0, len(encoded_mask), 2):
                    start_index = encoded_mask[i]

                    num_ones = encoded_mask[i + 1]

                    for j in range(start_index, start_index + num_ones - 1):
                        bitmap_flat[j] = 1
                # NOTE ordering of pixels in encoding is top -> down, then left -> right, hence 'F'
                bitmap = np.reshape(bitmap_flat, (im_h, im_w), order='F')
                result[defect_id] = bitmap
            else:
                result[defect_id] = None

        return result
