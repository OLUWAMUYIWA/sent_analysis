import numpy as np
import statistics as s
from baseline_trainer_SL import EMOTIONS_DICT

EMOTIONS = ['happy', 'angry', 'sad', 'others']
EMOTIONS_DICT = {'happy': 0,
            'angry': 1,
            'sad': 2,
            'others': 3}
NUM_EMO = len(EMOTIONS)

# data_path = 'data/train.txt'
def to_categorical(vec):
    to_ret = np.zeros((vec.shape[0], NUM_EMO))
    for idx, val in enumerate(vec):
        to_ret[idx, val] = 1
    return to_ret


def load_dev_labels(data_path='data/dev.txt'):
    CONV_PAD_LEN = 3
    target_list = []
    f = open(data_path, 'r', encoding='utf8')
    data_lines = f.readlines()
    f.close()

    for i, text in enumerate(data_lines):
        # first line is the name of the columns. ignore
        if i == 0:
            continue
        tokens = text.split('\t')
        emo = tokens[CONV_PAD_LEN + 1].strip()
        target_list.append(EMOTIONS_DICT[emo])
    ret = np.asarray(target_list)
    return ret

def get_metrics(ground, predictions):
    """Given predicted labels and the respective ground truth labels, display some metrics
        Input: shape [number of samples, NUM_CLASSES]
            predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
            ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
        Output:
            accuracy : Average accuracy
            microPrecision : Precision calculated on a micro level. Ref -
            https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
            microRecall : Recall calculated on a micro level
            microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions)
    ground = to_categorical(ground)
    true_positives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", true_positives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    #  Macro level calculation
    macro_precision = 0
    macroRecall = 0
    f1_list = []
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(NUM_EMO-1):
        precision = true_positives[c] / (true_positives[c] + falsePositives[c])
        macro_precision += precision
        recall = true_positives[c] / (true_positives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (EMOTIONS[c], precision, recall, f1))
    print('Harmonic Mean: ',
          s.harmonic_mean(f1_list))

    macro_precision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macro_precision) / (macro_precision + macroRecall) \
        if (macro_precision + macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
    macro_precision, macroRecall, macroF1))

    # Micro level calculation
    true_positives = true_positives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d"
          % (true_positives, falsePositives, falseNegatives))

    microPrecision = true_positives / (true_positives + falsePositives)
    microRecall = true_positives / (true_positives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall)\
        if (microPrecision + microRecall) > 0 else 0

    # predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (
    accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1

