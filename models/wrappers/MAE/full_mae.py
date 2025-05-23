from models.wrappers.MAE.core_mae import CoreMAE
from dataset.dataset_interface import EnumDatasetTaskType
from timm.models.layers import trunc_normal_
from optimizer import layer_decay

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

class FullMAE(CoreMAE):

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
            _, backbone_model = self.load_backbone(self._m_model)
            # Interpolate position embedding
            self.interpolate_pos_embed(self._m_model, backbone_model)
            trunc_normal_(self._m_model.head.weight, std=0.01)
        
        self.setup_optimizer(self._m_model)
        self.distribute()
        self.setup_criterion(in_dataset_task_type)
        self.load_checkpoint()

    def setup_optimizer(self, in_model):
        param_groups = layer_decay.param_groups_lrd(
            in_model,
            self._m_config.SOLVER.WEIGHT_DECAY,
            no_weight_decay_list=in_model.no_weight_decay(),
            layer_decay=self._m_config.MODEL.LAYER_DECAY,
        )
        return super().setup_optimizer(param_groups)
