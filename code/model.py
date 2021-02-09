from constants import *
from transformers import TFTFLxmertModel
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Input, Dense, LayerNormalization


def LxmertForQuestionAnswering():

    input_ids = Input(shape=(SEQ_LENGTH,))
    attention_mask = Input(shape=(SEQ_LENGTH,))
    visual_feats = Input(shape=(NUM_VISUAL_FEATURES, VISUAL_FEAT_DIM))
    normalized_boxes = Input(shape=(NUM_VISUAL_FEATURES, VISUAL_FEAT_DIM))
    token_type_ids = Input(shape=(SEQ_LENGTH,))

    lxmert = TFLxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')

    lxmert_output = lxmert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           visual_feats=visual_feats,
                           visual_pos=normalized_boxes,
                           token_type_ids=token_type_ids,
                           training=True
                           )

    last_hidden_states = lxmert_output.last_hidden_state

    x = Dense(1536, activation=gelu)(last_hidden_states)
    x = LayerNormalization(epsilon=1e-12)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=[input_ids, attention_mask, visual_feats,
                          normalized_boxes, token_type_ids], outputs=output)

    return model


def Train():
    train_generator = NotImplemented
    val_generator = NotImplemented

    model = LxmertForQuestionAnswering()
    optimizer = NotImplemented
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
