import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score, classification_report

# https://github.com/multitel-ai/urban-sound-tagging/blob/master/utils/metrics.py
def auprc(y_true, y_scores):
    """ Compute AUPRC for 1 class
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auc (float): the Area Under the Recall Precision curve
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    recall = np.concatenate((np.array([1.0]), recall, np.array([0.0])))
    precision = np.concatenate((np.array([0.0]), precision, np.array([1.0])))
    return auc(recall, precision)

def binary_confusion_matrix(y_true, y_scores):
    TN, FP, FN, TP = confusion_matrix(y_true, y_scores).ravel()
    return TN, FP, FN, TP

def compute_macro_auprc(y_true, y_scores, return_auprc_per_class=False):
    """ Compute macro AUPRC
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the macro AUPRC
    """
    _, num_classes = y_true.shape
    auprc_scores = [auprc(y_true[:,i],y_scores[:,i]) for i in range(num_classes)]
    # nanmean to ignore nan for borderline cases
    auprc_macro = np.nanmean(np.array(auprc_scores))
    if return_auprc_per_class:
        return auprc_scores, auprc_macro
    else:
        return auprc_macro

def compute_micro_auprc(y_true, y_scores):
    """ Compute micro AUPRC
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the micro AUPRC
    """
    auprc_micro = auprc(y_true.flatten(), y_scores.flatten())
    return auprc_micro

def compute_micro_F1(y_true, y_scores):
    """ Compute micro F1 @ 0.5
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            micro_F1 (float): the micro F1
    """
    micro_F1 = f1_score(y_true.flatten(), np.around(y_scores).flatten())
    return micro_F1

def metric_fn_sonycust(Y_true, Y_mask, preds):
    # Get classwise auprc
    avg_auprc = []
    for i in range(labels.shape[-1]):
        labels = Y_true[:, i]
        labels_mask = Y_mask[:, i]
        predictions = preds[:, i]

        # Get relevant indices from the mask
        relevant_inds = np.where(labels_mask)[0]
        
        # get AUC before thresholding
        aupr = auprc(labels[relevant_inds], predictions[relevant_inds])

        avg_auprc.append(aupr)
        
    metrics = {
        'auprc_macro': np.array(avg_auprc),
        'F1_micro': compute_micro_F1(Y_true[Y_mask], preds[Y_mask])
        'auprc_micro': compute_micro_F1(Y_true[Y_mask], preds[Y_mask])
    }
    return metrics

def metric_fn_openmic(Y_true, Y_mask, preds, threshold=0.5):

    # Get classwise classification metrics
    avg_fscore_weighted = []
    avg_fscore_macro = []
    avg_precision_macro = []
    avg_recall_macro = []
    auroc = []

    for i in range(labels.shape[-1]):
        labels = Y_true[:, i]
        labels_mask = Y_mask[:, i]
        predictions = preds[:, i]

        # Get relevant indices from the mask
        relevant_inds = np.where(labels_mask)[0]
        
        # get AUC before thresholding
        auc = roc_auc_score(labels[relevant_inds], predictions[relevant_inds])

        # Binarize the predictions based on the threshold.
        predictions[predictions >= threshold] = 1
        predictions[predictions < 1] = 0
#     print(classification_report(
#         labels[relevant_inds], predictions[relevant_inds]))
    # return classification report
        results = classification_report(labels[relevant_inds], predictions[relevant_inds]

        avg_fscore_weighted.append(results['weighted avg']['f1-score'])
        avg_fscore_macro.append(results['macro avg']['f1-score'])
        avg_precision_macro.append(results['macro avg']['precision'])
        avg_recall_macro.append(results['macro avg']['recall'])
        auroc.append(auc)

    metrics = {
        'F1_macro': np.array(avg_fscore_macro),
        'F1_weighted': np.array(avg_fscore_weighted),
        'precision': np.array(avg_precision_macro),
        'recall': np.array(avg_recall_macro),
        'auroc': np.array(auroc)
    }
    return metrics