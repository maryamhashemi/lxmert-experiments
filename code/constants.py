import os

BASE_PATH = '/home/ubuntu/hashemi/LXMERT'

QA_TRAIN_PATH = os.path.join(BASE_PATH, 'data/train.json')
QA_VAL_PATH = os.path.join(BASE_PATH, 'data/...')

TRAIN_IMGFEAT = os.path.join(
    BASE_PATH, 'data/mscoco_imgfeat/train2014_obj36.tsv')
VAL_IMGFEAT = os.path.join(BASE_PATH, 'data/mscoco_imgfeat/val2014_obj36.tsv')

ANS2LABELS_PATH = os.path.join(BASE_PATH, 'data/trainval_ans2label.json')

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

SEQ_LENGTH = 20
EPOCHS = 4
BATCH_SIZE = 32
NUM_CLASSES = NotImplemented
NUM_VISUAL_FEATURES = NotImplemented
VISUAL_FEAT_DIM = NotImplemented
