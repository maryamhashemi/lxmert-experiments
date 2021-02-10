from constants import *
from transformers import TFLxmertModel
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam


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
    output = Dense(NUM_CLASSES, activation='linear')(x)

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
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
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
