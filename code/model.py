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

    input_ids = Input(shape=(SEQ_LENGTH,))
    attention_mask = Input(shape=(SEQ_LENGTH,))
    visual_feats = Input(shape=(NUM_VISUAL_FEATURES, VISUAL_FEAT_DIM))
    normalized_boxes = Input(shape=(NUM_VISUAL_FEATURES, VISUAL_POS_DIM))

    lxmert = TFLxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')

    lxmert_output = lxmert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           visual_feats=visual_feats,
                           visual_pos=normalized_boxes,
                           training=True
                           )

    last_hidden_states = lxmert_output.last_hidden_state

    x = Dense(1536, activation=gelu)(last_hidden_states)
    x = LayerNormalization(epsilon=1e-12)(x)
    output = Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=[input_ids, attention_mask, visual_feats,
                          normalized_boxes], outputs=output)

    return model


def Train():
    logger.info("start loading %s", % (TRAIN_QA_PATH))
    train_img_ids, train_ques_ids, train_ques_inputs, train_labels = get_QA(
        TRAIN_QA_PATH)

    logger.info("start loading %s", % (VAL_QA_PATH))
    val_img_ids, val_ques_ids, val_ques_inputs, val_labels = get_QA(
        VAL_QA_PATH)

    train_generator = DataGenerator(train_img_ids,
                                    train_ques_ids,
                                    train_ques_inputs,
                                    train_labels,
                                    TRAIN_IMGFEAT_PATH,
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
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x=train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        workers=6,
                        use_multiprocessing=True)

    # save history
    with open(HISTORY_PATH, 'w') as file:
        json.dump(history.history, file)

    return


def fit(model, train_generator, val_generator, loss_fn, optimizer, quesid2data):

    for epoch in range(EPOCHS):
        logger.info("\nStart of epoch %d" % (epoch,))
        quesid2ans = {}
        # Iterate over the batches of the dataset.
        for step, (ques_ids, x_batch_train, y_batch_train) in enumerate(train_generator):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                # Logits for this minibatch
                logits = model(x_batch_train, training=True)

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            score, label = logits.max(1)
            for qid, l in zip(ques_id, label.numpy()):
                ans = LABEL2ANS[l]
                quesid2ans[qid.item()] = ans

            # Log every 200 batches.
            if step % 100 == 0:
                logger.info(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
        logger.info("\nEpoch %d: Train %0.2f\n" %
                    (epoch, evaluate(quesid2ans, quesid2data) * 100.))

        # Run a validation loop at the end of each epoch.
        quesid2ans = {}
        for ques_id, x_batch_val, y_batch_val in val_generator:
            val_logits = model(x_batch_val, training=False)
            score, label = val_logits.max(1)
            for qid, l in zip(ques_id, label.numpy()):
                ans = LABEL2ANS[l]
                quesid2ans[qid.item()] = ans

        logger.info("\nEpoch %d: Val %0.2f\n" %
                    (epoch, evaluate(quesid2ans, quesid2data) * 100.))

    return


def evaluate(quesid2ans, quesid2data):
    score = 0.
    for quesid, ans in quesid2ans.items():
        data = quesid2data[quesid]
        label = data['label']
        if ans in label:
            score += label[ans]
    return score / len(quesid2ans)
