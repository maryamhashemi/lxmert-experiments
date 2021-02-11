import json
import logging
import numpy as np
import pandas as pd
from constants import *
from transformers import LxmertTokenizer

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('prepare_QA.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_QA(QAdata_path):
    """
    load question and answers.

    Arguments:
    QAdata_path -- a string that shows questions and answers file path.

    Return:
    img_ids -- a list of image ids.
    ques_inputs -- a list of question ids.
    inputs -- tokenized questions and language attention masks.
    targets -- labels

    """
    data = []
    for path in QAdata_path:
        data += json.load(open(path))

    quesid2data = {
        d['question_id']: d
        for d in data
    }

    data = pd.DataFrame(data)
    logger.info("successfully load questions data.")

    img_ids = data['img_id'].values
    ques_ids = data['question_id'].values
    questions = list(data['sent'].values)
    labels = data['label'].values
    assert len(img_ids) == len(ques_ids) == len(questions) == len(labels)

    # Tokenize question
    lxmert_tokenizer = LxmertTokenizer.from_pretrained(
        "unc-nlp/lxmert-base-uncased")

    ques_inputs = lxmert_tokenizer(questions,
                                   padding="max_length",
                                   max_length=SEQ_LENGTH,
                                   truncation=True,
                                   return_attention_mask=True,
                                   add_special_tokens=True,
                                   return_tensors="tf"
                                   )
    # Provide label (target)
    ans2label = json.load(open(ANS2LABELS_PATH))
    num_answers = len(ans2label)

    targets = np.zeros((len(labels), num_answers))
    for i, label in enumerate(labels):
        for ans, score in label.items():
            targets[i, ans2label[ans]] = score

    logger.info("total number of img_ids is %i ." % (len(img_ids)))
    logger.info("total number of ques_ids is %i ." % (len(ques_ids)))
    logger.info("total number of ques_inputs is %s ." %
                (str(ques_inputs.input_ids.shape)))
    logger.info("total number of labels is %s ." % (str(targets.shape)))

    return img_ids, ques_ids, ques_inputs, targets, quesid2data


# img_ids, ques_ids, ques_inputs, labels, quesid2data = get_QA(
#     [MINIVAL_QA_PATH])
