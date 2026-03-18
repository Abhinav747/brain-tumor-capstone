from sklearn.metrics import confusion_matrix
import numpy as np

def compute_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    tnr = tn/(tn+fp)

    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)

    f1 = 2*(precision*recall)/(precision+recall)

    f2 = (5*precision*recall)/(4*precision+recall)

    jaccard = tp/(tp+fp+fn)

    hamming = (fp+fn)/(tp+tn+fp+fn)

    return {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "tnr":tnr,
        "fpr":fpr,
        "fnr":fnr,
        "f1":f1,
        "f2":f2,
        "jaccard":jaccard,
        "hamming":hamming
    }