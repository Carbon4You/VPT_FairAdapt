from enum import Enum

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

class StepType(Enum):
    TRAINING = 0
    VALIDATING = 1
    TESTING = 2

    @staticmethod
    def from_str(label):
        if label in ("TRAINING"):
            return StepType.TRAINING
        elif label in ("VALIDATING"):
            return StepType.VALIDATING
        elif label in ("TESTING"):
            return StepType.TESTING
        else:
            raise NotImplementedError

    def to_str(label):
        if label == StepType.TRAINING:
            return "TRAINING"
        elif label == StepType.VALIDATING:
            return "VALIDATING"
        elif label == StepType.TESTING:
            return "TESTING"
        else:
            raise NotImplementedError
