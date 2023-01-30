from munkres import Munkres
import numpy as np


def hungarian(gt, pred, num_classes=50):
    matrix = np.zeros(shape=(num_classes, num_classes), dtype=np.float32)
    preds = {}
    gts = {}
    for i in range(num_classes):
        preds[i] = np.array([i in item for item in pred])
        gts[i] = np.array([i in item for item in gt])

    for i in range(num_classes):
        for j in range(num_classes):
            coi = np.logical_and(preds[i], gts[j])
            matrix[i][j] = 1.0 - 1.0 * np.sum(coi) / len(gt)

    matrix = matrix.tolist()
    m = Munkres()
    indexes = m.compute(matrix)
    trans = {}
    for row, column in indexes:
        trans[row + 1] = column + 1

    return trans
