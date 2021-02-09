import sys
import csv
import pickle
import base64
import logging
import numpy as np
from constants import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('prepare_imgfeat.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

csv.field_size_limit(sys.maxsize)


def load_obj_tsv(fname, save_dir):
    """
    Load object features from tsv file and save object features in seperated files.

    Arguments:

    fname -- The path to the tsv file.
    save_dir -- The path to the saved directory.
    """

    logger.info("Start to load Faster-RCNN detected objects from %s" % fname)

    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")

        for i, item in enumerate(reader):
            new_item = {}

            num_boxes = int(item['num_boxes'])
            img_h, img_w = int(item['img_h']), int(item['img_w'])
            name = str(item['img_id'])

            boxes = np.frombuffer(base64.b64decode(
                item['boxes']), dtype=np.float32)
            boxes = boxes.reshape((num_boxes, 4))
            copy_boxes = boxes.copy()

            # Normalize the boxes (to 0 ~ 1)
            copy_boxes[:, (0, 2)] /= img_w
            copy_boxes[:, (1, 3)] /= img_h

            feats = np.frombuffer(base64.b64decode(
                item['features']), dtype=np.float32)
            feats = feats.reshape((num_boxes, -1))

            new_item['boxes'] = copy_boxes
            new_item['features'] = feats

            with open(save_dir + name + '.pkl', 'wb') as f:
                pickle.dump(new_item, f, pickle.HIGHEST_PROTOCOL)

    logger.info("saved images.")
    return


load_obj_tsv(TRAIN2014_OBJ36_PATH, TRAIN_IMGFEAT_PATH)
load_obj_tsv(VAL2014_OBJ36_PATH, VAL_IMGFEAT_PATH)
