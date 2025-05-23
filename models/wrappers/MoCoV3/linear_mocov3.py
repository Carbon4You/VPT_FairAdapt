from models.wrappers.MoCoV3.core_mocov3 import CoreMoCoV3
from dataset.dataset_interface import EnumDatasetTaskType

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

class LinearMoCoV3(CoreMoCoV3):

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
            # Initialize the FC layer.
            self._m_model.head.weight.data.normal_(mean=0.0, std=0.01)
            self._m_model.head.bias.data.zero_()
        
        # Freeze all but the head
        for _, p in self._m_model.named_parameters():
            p.requires_grad = False
        for _, p in self._m_model.head.named_parameters():
            p.requires_grad = True
        parameters = list(filter(lambda p: p.requires_grad, self._m_model.parameters()))
        assert len(parameters) == 2
        
        self._m_optimizer = self.setup_optimizer(self._m_model.head.parameters())
        self.setup_criterion(in_dataset_task_type)
        self.distribute()
        self.load_checkpoint()

    def _f_save_model(self, in_start_epoch, in_last_epoch, in_results, in_model_name, in_model, in_optimizer, in_config, in_output):
        return super()._f_save_model(in_start_epoch, in_last_epoch, in_results, in_model_name, in_model, in_optimizer, in_config, in_output)
