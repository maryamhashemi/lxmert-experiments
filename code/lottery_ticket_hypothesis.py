import logging
from constants import *
from prepare_QA import *
from data_generator import *
from model import *
import tensorflow as tf
from transformers import TFLxmertModel, LxmertForQuestionAnswering
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('lottery_ticket_hypothesis.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def weight_prune_layer(k_weights, k_sparsity):
    """
    Takes in matrices of kernel and bias weights (for a dense
      layer) and returns the unit-pruned versions of each
    Args:
      k_weights:
      k_sparsity: percentage of weights to set to 0
    Returns:
      kernel_weights: sparse matrix with same shape as the original
        kernel weight matrix
    """

    # Copy the kernel weights and get ranked indeces of the abs
    kernel_weights = np.copy(k_weights)
    sorterd_ind = np.argsort(np.abs(kernel_weights), axis=None)
    nonzero_ind = np.flatnonzero(kernel_weights)
    ind = np.unravel_index(np.intersect1d(
        sorterd_ind, nonzero_ind), kernel_weights.shape)

    # Number of indexes to set to 0
    cutoff = int(len(ind[0])*k_sparsity)
    # The indexes in the 2D kernel weight matrix to set to 0
    if len(kernel_weights.shape) == 2:
        sparse_cutoff_inds = (ind[0][0:cutoff], ind[1][0:cutoff])
    if len(kernel_weights.shape) == 1:
        sparse_cutoff_inds = (ind[0][0:cutoff])
    kernel_weights[sparse_cutoff_inds] = 0.

    return kernel_weights


def get_pruned_weights(model, model_score, val_generator, val_quesid2data):
    weights = model.get_weights()
    not_pruned_layers = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]

    while(True):
        newWeightList = []
        for i in range(0, len(weights)):
            if i in not_pruned_layers and i == len(weights):
                newWeightList.append(weights[i])
            else:
                kernel_weights = weight_prune_layer(weights[i], 0.01)
                newWeightList.append(kernel_weights)

        model.set_weights(newWeightList)
        score = get_val_score(val_generator, model, val_quesid2data)

        weights = newWeightList.copy()

        if(score < 0.9*model_score):
            break

    model.set_weights(weights)
    model.save_weights("pruned_weights.h5")
    logger.info("successfully save pruned weights.")

    return weights


def prepare_init_weights(pruned_weights, init_weights):
    init_pruned_weights = []

    for i in range(0, len(pruned_weights)):
        new_weight = np.where(
            np.array(pruned_weights[i]) > 0, init_weights[i], pruned_weights[i])
        init_pruned_weights.append(new_weight)

    return init_pruned_weights


def get_val_score(val_generator, model, val_quesid2data):
    quesid2ans = {}
    for ques_ids, x_batch_val, y_batch_val in val_generator:
        val_logits = val_step(x_batch_val, y_batch_val, model)

        label = tf.argmax(val_logits, axis=1)
        for qid, l in zip(ques_ids, label):
            ans = LABEL2ANS[l]
            quesid2ans[qid] = ans

    score = evaluate(quesid2ans, val_quesid2data) * 100.
    logger.info("Val accuracy: %0.2f\n" % (score))

    return score


def retrain(loading_weights_path=None,
            init_weights_path=None,
            saving_weights_path=None):

    logger.info("start loading %s and %s" % (TRAIN_QA_PATH, NOMINIVAL_QA_PATH))
    train_img_ids, train_ques_ids, train_ques_inputs, train_labels, train_quesid2data = get_QA(
        [TRAIN_QA_PATH, NOMINIVAL_QA_PATH])

    logger.info("start loading %s" % (MINIVAL_QA_PATH))
    val_img_ids, val_ques_ids, val_ques_inputs, val_labels, val_quesid2data = get_QA([
                                                                                     MINIVAL_QA_PATH])

    train_generator = DataGenerator(train_img_ids,
                                    train_ques_ids,
                                    train_ques_inputs,
                                    train_labels,
                                    [TRAIN_IMGFEAT_PATH, VAL_IMGFEAT_PATH],
                                    BATCH_SIZE)
    logger.info("successfully build train generator.")

    val_generator = DataGenerator(val_img_ids,
                                  val_ques_ids,
                                  val_ques_inputs,
                                  val_labels,
                                  [VAL_IMGFEAT_PATH],
                                  BATCH_SIZE,
                                  False)
    logger.info("successfully build val generator.")

    model = TFLxmertForQuestionAnswering()

    if (loading_weights_path is not None):
        model.load_weights(loading_weights_path)

    score = get_val_score(val_generator, model, val_quesid2data)

    pruned_weights = get_pruned_weights(
        model, score, val_generator, val_quesid2data)
    logger.info("successfully prune model.")

    if (init_weights_path is not None):
        model.load_weights(init_weights_path)
    init_weights = model.get_weights()
    logger.info("successfully get init weights of full model.")

    init_pruned_weights = prepare_init_weights(pruned_weights, init_weights)
    logger.info("successfully get init weights for pruned model.")

    model.set_weights(init_pruned_weights)
    optimizer = Adam(learning_rate=LR)
    loss_fn = BinaryCrossentropy(from_logits=True)
    logger.info("start training the  pruned model.")
    fit(model, train_generator, val_generator, loss_fn,
        optimizer, train_quesid2data, val_quesid2data, saving_weights_path)

    return


def check():
    logger.info("start loading %s" % (MINIVAL_QA_PATH))
    val_img_ids, val_ques_ids, val_ques_inputs, val_labels, val_quesid2data = get_QA([
                                                                                     MINIVAL_QA_PATH])

    val_generator = DataGenerator(val_img_ids,
                                  val_ques_ids,
                                  val_ques_inputs,
                                  val_labels,
                                  [VAL_IMGFEAT_PATH],
                                  BATCH_SIZE,
                                  False)
    logger.info("successfully build val generator.")

    model = TFLxmertForQuestionAnswering()

    model.load_weights("initial_weights.h5")
    init_weights = model.get_weights()
    logger.info("successfully get init weights of full model.")

    model.load_weights("pruned_weights.h5")
    pruned_weights = model.get_weights()
    logger.info("successfully get pruned weights.")

    init_pruned_weights = prepare_init_weights(pruned_weights, init_weights)
    logger.info("successfully get init weights for pruned model.")

    model.set_weights(init_pruned_weights)
    score = get_val_score(val_generator, model, val_quesid2data)

    return

# retrain(loading_weights_path="fine_tuning_LXMERT_4.h5",
#         init_weights_path="initial_weights.h5",
#         saving_weights_path="fine_tuning_pruned_LXMERT.h5")


check()
