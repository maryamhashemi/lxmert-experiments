import pickle
from os import path
import logging
import numpy as np
from constants import *
from tensorflow.keras.utils import Sequence


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('data_generator.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class DataGenerator(Sequence):

    def __init__(self, img_ids, ques_ids, ques_inputs, labels, imgfeat_path, batch_size, shuffle=True):

        self.img_ids = img_ids
        self.ques_ids = ques_ids
        self.ques_inputs = ques_inputs
        self.labels = labels
        self.imgfeat_path = imgfeat_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        """
        return int(np.ceil(self.ques_ids.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data

        """
        # Generate indexes of the batch
        if ((index+1)*self.batch_size <= self.ques_ids.shape[0]):
            indexes = self.indexes[index *
                                   self.batch_size:(index + 1) * self.batch_size]
        else:
            indexes = self.indexes[index * self.batch_size:]

        # Generate data
        ques_ids, [input_ids, attention_masks, visual_feats,
                   normalized_boxes], labels = self.__data_generation(indexes)

        logger.info("get %i/%i batches of data." % (index+1, self.__len__()))
        return ques_ids, [input_ids, attention_masks, visual_feats, normalized_boxes], labels

    def on_epoch_end(self):
        """
        Updates indexes after each epoch'
        """
        self.indexes = np.arange(self.ques_ids.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        logger.info("end of epoch and shuffle data.")

    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples
        """
        batch = indexes.shape[0]

        ques_ids = np.empty((batch,))
        input_ids = np.empty((batch, SEQ_LENGTH))
        attention_masks = np.empty((batch, SEQ_LENGTH))
        visual_feats = np.empty((batch, NUM_VISUAL_FEATURES, VISUAL_FEAT_DIM))
        normalized_boxes = np.empty(
            (batch, NUM_VISUAL_FEATURES, VISUAL_POS_DIM))
        labels = np.empty((batch, NUM_CLASSES))

        for i, idx in enumerate(indexes):
            # Store sample
            ques_ids[i] = self.ques_ids[idx]
            input_ids[i] = self.ques_inputs.input_ids[idx]
            attention_masks[i] = self.ques_inputs.input_ids[idx]

            for imgfeat_path in self.imgfeat_path:
                if path.exists(imgfeat_path + str(self.img_ids[idx]) + '.pkl'):
                    with open(imgfeat_path + str(self.img_ids[idx]) + '.pkl', 'rb') as f:
                        image = pickle.load(f)
                    break

            visual_feats[i] = image['features']
            normalized_boxes[i] = image['boxes']

            # Store class
            labels[i] = self.labels[idx]

        logger.info("successfully create one batch of data.")
        return ques_ids, [input_ids, attention_masks, visual_feats, normalized_boxes], labels
