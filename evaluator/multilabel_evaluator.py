import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import numpy as np
from typing import Tuple
from models.enum_step_types import StepType

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

class MultilabelEvaluator:
    def __init__(self, in_writer: SummaryWriter) -> None:
        self._m_writer = in_writer

    def __call__(
        self,
        in_step_type: StepType,
        in_epoch: int,
        in_target: np.ndarray,
        in_output: np.ndarray,
    ) -> dict:
        ap, ar, mAP, mAR = self.compute_precision_recall(
            in_step_type, in_epoch, in_target, in_output
        )

        auc, m_auc = self.compute_auc(in_step_type, in_epoch, in_target, in_output)

        accuracy, m_accuracy = self.compute_accuracy(
            in_step_type, in_epoch, in_target, in_output
        )

        results = {
            "mAP": mAP,
            "mAR": mAR,
            "mAUC": m_auc,
            "m_Accuracy": m_accuracy,
            "ap": ap,
            "ar": ar,
            "auc": auc,
            "accuracy": accuracy,
        }

        return results

    def compute_precision_recall(
        self,
        in_step_type: StepType,
        in_epoch: int,
        in_target: np.ndarray,
        in_output: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        nb_classes = in_output.shape[1]

        ap = np.zeros((nb_classes,), dtype=np.float32)
        ar = np.zeros((nb_classes,), dtype=np.float32)

        for class_index in range(nb_classes):
            y_true = in_target[:, class_index]
            y_pred = in_output[:, class_index]

            try:
                ap[class_index] = average_precision_score(y_true, y_pred)
            except ValueError:
                ap[class_index] = -1

            metric_name = in_step_type.to_str() + "_AP_C" + str(class_index)
            self._m_writer.add_scalar(metric_name, ap[class_index], in_epoch)

            try:
                _, rec, _ = precision_recall_curve(y_true, y_pred)
                ar[class_index] = rec.mean()
            except ValueError:
                ar[class_index] = -1

            metric_name = in_step_type.to_str() + "_AR_C" + str(class_index)
            self._m_writer.add_scalar(metric_name, ar[class_index], in_epoch)

        mAP = ap.mean()

        metric_name = in_step_type.to_str() + "_mAP"
        self._m_writer.add_scalar(metric_name, mAP, in_epoch)

        mAR = ar.mean()

        metric_name = in_step_type.to_str() + "_mAR"
        self._m_writer.add_scalar(metric_name, mAR, in_epoch)

        return ap, ar, mAP, mAR

    def compute_auc(
        self,
        in_step_type: StepType,
        in_epoch: int,
        in_target: np.ndarray,
        in_output: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:

        nb_classes = in_output.shape[1]

        aucs = np.zeros((nb_classes,), dtype=np.float32)

        target = torch.Tensor(in_target.copy())
        output = torch.sigmoid(torch.Tensor(in_output.copy()))

        for class_index in range(nb_classes):
            y_true = target[:, class_index]
            y_pred = output[:, class_index]
            try:
                auc = roc_auc_score(y_true, y_pred)
            except ValueError:
                auc = -1
            aucs[class_index] = auc

            metric_name = in_step_type.to_str() + "_AUC_C" + str(class_index)
            self._m_writer.add_scalar(metric_name, auc, in_epoch)

        mAUC = aucs.mean()
        self._m_writer.add_scalar(in_step_type.to_str() + "_mAUC", mAUC, in_epoch)
        return aucs, mAUC

    def compute_accuracy(
        self,
        in_step_type: StepType,
        in_epoch: int,
        in_target: np.ndarray,
        in_output: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:

        nb_classes = in_output.shape[1]

        accuracies = np.zeros((nb_classes,), dtype=np.float32)

        target = torch.Tensor(in_target.copy())
        output = torch.round(torch.sigmoid(torch.Tensor(in_output.copy())))

        for class_index in range(nb_classes):
            y_true = target[:, class_index]
            y_pred = output[:, class_index]
            try:
                acc = accuracy_score(y_true, y_pred)
            except ValueError:
                acc = -1
            accuracies[class_index] = acc

            metric_name = in_step_type.to_str() + "_ACC_C" + str(class_index)
            self._m_writer.add_scalar(metric_name, acc, in_epoch)

        mAccuracy = accuracies.mean()
        self._m_writer.add_scalar(in_step_type.to_str() + "_mACC", mAccuracy, in_epoch)
        return accuracies, mAccuracy
