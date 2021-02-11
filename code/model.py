import logging
from constants import *
import tensorflow as tf
from transformers import TFLxmertModel
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from data_generator import *
from prepare_QA import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('model.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def LxmertForQuestionAnswering():

    input_ids = Input(shape=(SEQ_LENGTH,), dtype=tf.int32)
    attention_mask = Input(shape=(SEQ_LENGTH,))
    visual_feats = Input(shape=(NUM_VISUAL_FEATURES, VISUAL_FEAT_DIM))
    normalized_boxes = Input(shape=(NUM_VISUAL_FEATURES, VISUAL_POS_DIM))

    lxmert = TFLxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')

    lxmert_output = lxmert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           visual_feats=visual_feats,
                           visual_pos=normalized_boxes,
                           return_dict=True,
                           training=True)

    last_hidden_states = lxmert_output.pooled_output

    x = Dense(1536, activation=gelu)(last_hidden_states)
    x = LayerNormalization(epsilon=1e-12)(x)
    output = Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=[input_ids, attention_mask, visual_feats,
                          normalized_boxes], outputs=output)

    return model


def Train():
    logger.info("start loading %s and %s" % (TRAIN_QA_PATH, NOMINIVAL_QA_PATH))
    train_img_ids, train_ques_ids, train_ques_inputs, train_labels, train_quesid2data = get_QA(
        [TRAIN_QA_PATH, NOMINIVAL_QA_PATH])

    logger.info("start loading %s" % (MINIVAL_QA_PATH))
    val_img_ids, val_ques_ids, val_ques_inputs, val_labels, val_quesid2data = get_QA(
        [MINIVAL_QA_PATH])

    train_generator = DataGenerator(train_img_ids,
                                    train_ques_ids,
                                    train_ques_inputs,
                                    train_labels,
                                    [TRAIN_IMGFEAT_PATH, VAL_IMGFEAT_PATH],
                                    BATCH_SIZE)
    logger.info("successfully build train generator")

    val_generator = DataGenerator(val_img_ids,
                                  val_ques_ids,
                                  val_ques_inputs,
                                  val_labels,
                                  VAL_IMGFEAT_PATH,
                                  BATCH_SIZE,
                                  False)
    logger.info("successfully build val generator")

    model = LxmertForQuestionAnswering()
    optimizer = Adam(learning_rate=LR)
    loss_fn = binary_crossentropy

    model.summary()
    fit(model, train_generator, val_generator, loss_fn,
        optimizer, train_quesid2data, val_quesid2data)

    return


def fit(model, train_generator, val_generator, loss_fn, optimizer, train_quesid2data, val_quesid2data):

    for epoch in range(EPOCHS):
        logger.info("\nStart of epoch %d" % (epoch,))
        quesid2ans = {}
        # Iterate over the batches of the dataset.
        for step, (ques_ids, x_batch_train, y_batch_train) in enumerate(train_generator):
            logits, loss_value = train_step(x_batch_train,
                                            y_batch_train,
                                            model,
                                            loss_fn,
                                            optimizer)
            label = tf.argmax(logits, axis=1)
            for qid, l in zip(ques_ids, label):
                ans = LABEL2ANS[l]
                quesid2ans[qid.item()] = ans

            if step % 100 == 0:
                logger.info("\nstep %d" % (step))

        logger.info("\nEpoch %d: Train %0.2f\n" %
                    (epoch, evaluate(quesid2ans, train_quesid2data) * 100.))

        # Run a validation loop at the end of each epoch.
        quesid2ans = {}
        for ques_ids, x_batch_val, y_batch_val in val_generator:
            val_logits = val_step(x_batch_train, y_batch_train, model)

            score, label = tf.argmax(val_logits, axis=1)
            for qid, l in zip(ques_ids, label):
                ans = LABEL2ANS[l]
                quesid2ans[qid.item()] = ans

        logger.info("\nEpoch %d: Val %0.2f\n" %
                    (epoch, evaluate(quesid2ans, val_quesid2data) * 100.))

    model.save_weights('fine_tuning_LXMERT.h5')
    logger.info("\n save model weights into fine_tuning_LXMERT.h5")
    return


def evaluate(quesid2ans, quesid2data):
    score = 0.
    for quesid, ans in quesid2ans.items():
        data = quesid2data[quesid]
        label = data['label']
        if ans in label:
            score += label[ans]
    return score / len(quesid2ans)


@tf.function
def train_step(x, y, model, loss_fn, optimizer):

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return logits, loss_value


@tf.function
def val_step(x, y, model):
    val_logits = model(x, training=False)
    return val_logits


Train()
