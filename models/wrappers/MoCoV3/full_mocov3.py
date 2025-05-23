from timm.models import create_model
from models.wrappers.MoCoV3.core_mocov3 import CoreMoCoV3
from dataset.dataset_interface import EnumDatasetTaskType

import utility.logger as logger

logger_handle = logger.get_logger("vpt_demographic_adaptation")


class FullMoCoV3(CoreMoCoV3):

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

    def initialize(
        self, in_num_of_classes: int, in_dataset_task_type: EnumDatasetTaskType, **kwargs
    ):
        super().initialize(in_num_of_classes, in_dataset_task_type, **kwargs)

        self._m_model = self.setup_model(in_num_of_classes)
        if self._m_config.SOLVER.FIRST_EPOCH == 0:
            self.load_backbone(self._m_model)
        self._m_optimizer = self.setup_optimizer(self._m_model.parameters())
        self.setup_criterion(in_dataset_task_type)
        self.distribute()
        self.load_checkpoint()

    def setup_model(self, in_num_of_classes):
        model = create_model(
            self._m_config.MODEL.ARCH,
            pretrained=False,
            num_classes=in_num_of_classes,
            drop_rate=self._m_config.MODEL.DROP_RATE,
            drop_path_rate=self._m_config.MODEL.DROP_RATE_PATH,
            img_size=self._m_config.DATASET.IMAGE_SIZE,
        )
        return model
