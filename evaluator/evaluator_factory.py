from evaluator.multilabel_evaluator import MultilabelEvaluator
from dataset.dataset_interface import EnumDatasetTaskType
from torch.utils.tensorboard import SummaryWriter

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

class NullWriter():
    def __init__(self, *args, **kwargs):
        pass
    def add_scalar(self, *args, **kwargs):
        pass
    def flush(self, *args, **kwargs):
        pass

class EvaluatorFactory:
    def get(in_dataset_task_type: EnumDatasetTaskType, in_writer: SummaryWriter):
        if in_dataset_task_type == EnumDatasetTaskType.MULTILABEL:
            return MultilabelEvaluator(in_writer)
        else:
            print(
                "ERROR : EvaluatorFactory().get(...) : DATASET TASK TYPE not supported."
            )
