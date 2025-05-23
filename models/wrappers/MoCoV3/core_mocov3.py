import torch
import pathlib

from models.wrappers.core_model import CoreModel
import models.wrappers.MoCoV3.moco_vit

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

class CoreMoCoV3(CoreModel):
    def __init__(
        self,
        in_config,
        in_distribution=None,
        in_device=None,
        in_gpu=None,
        in_global_rank=0,
        in_global_world_size=1,
    ) -> None:
        super().__init__(
            in_config,
            in_distribution,
            in_device,
            in_gpu,
            in_global_rank,
            in_global_world_size,
        )
        self._m_linear_keyword = "head"

    def setup_model(self, in_num_of_classes):
        self._m_model = models.wrappers.MoCoV3.moco_vit.__dict__[self._m_config.MODEL.ARCH](
            in_config=self._m_config,
            img_size=self._m_config.DATASET.IMAGE_SIZE,
            num_classes=in_num_of_classes,
        )
        return self._m_model

