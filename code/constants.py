import os
import json

BASE_PATH = '/home/ubuntu/hashemi/LXMERT'

TRAIN_QA_PATH = os.path.join(BASE_PATH, 'data/train.json')
NOMINIVAL_QA_PATH = os.path.join(BASE_PATH, 'data/nominival.json')
MINIVAL_QA_PATH = os.path.join(BASE_PATH, 'data/minival.json')

TRAIN2014_OBJ36_PATH = os.path.join(
    BASE_PATH, 'data/mscoco_imgfeat/train2014_obj36.tsv')
VAL2014_OBJ36_PATH = os.path.join(
    BASE_PATH, 'data/mscoco_imgfeat/val2014_obj36.tsv')

TRAIN_IMGFEAT_PATH = os.path.join(
    BASE_PATH, 'data/mscoco_imgfeat/train/')
VAL_IMGFEAT_PATH = os.path.join(
    BASE_PATH, 'data/mscoco_imgfeat/val/')

ANS2LABELS_PATH = os.path.join(BASE_PATH, 'data/trainval_ans2label.json')
LABELS2ANS_PATH = os.path.join(BASE_PATH, 'data/trainval_label2ans.json')

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

SEQ_LENGTH = 20
EPOCHS = 5
BATCH_SIZE = 32
LR = 5e-5
NUM_CLASSES = len(json.load(open(ANS2LABELS_PATH)))
NUM_VISUAL_FEATURES = 36
VISUAL_FEAT_DIM = 2048
VISUAL_POS_DIM = 4
HISTORY_PATH = NotImplemented
LABEL2ANS = json.load(open(LABELS2ANS_PATH))
