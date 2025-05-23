class ModelInterface:

    def __init__(
        self,
        in_config,
        in_distribution=None,
        in_device=None,
        in_gpu=None,
        in_global_rank=0,
        in_global_world_size=1,
    ) -> None:
        
        self._m_config = in_config
        self._m_gpu = in_gpu
        self._m_global_rank = in_global_rank
        self._m_global_world_size = in_global_world_size
        self._m_device = in_device
        self._m_distribution = in_distribution
        
        self._m_model = None
        self._m_optimizer = None
        self.save_checkpoint = self.null_save
        
    def initialize(self, in_num_of_classes, in_dataset_task_type, **kwargs):
        raise NotImplementedError()

    def distribute(self):
        raise NotImplementedError()

    def setup_model(self, *args, **kwargs):
        raise NotImplementedError()

    def load_backbone(self, *args, **kwargs):
        raise NotImplementedError()

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError()

    def setup_optimizer(self, *args, **kwargs):
        raise NotImplementedError()

    def optimizer(self):
        return self._m_optimizer

    def setup_criterion(self, in_task_type):
        raise NotImplementedError()

    # Steps

    def step(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()

    def null_evaluate(self, *args, **kwargs):
        return None

    def batch_backward(self, in_losses, in_optimizer):
        raise NotImplementedError()
    
    def null_batch_backward(self, *args, **kwargs):
        return None

    def null_save(self, *args, **kwargs):
        return None
    
    def save_model(self, in_start_epoch, in_last_epoch):
        raise NotImplementedError()

    def save_data(
        self, in_output_dir, in_step_type, in_epoch, in_targets, in_outputs,in_attribute
    ):
        raise NotImplementedError()

    # Helper functions

    def transmitter(self, in_target, in_output, in_attributes):
        raise NotImplementedError()

    def receiver(self, in_target, in_output, in_attributes):
        raise NotImplementedError()

    # Set/Get

    def model(self):
        return self._m_model

    def training_type(self):
        return self._m_training_type

    def get_train_transformer(self, *args, **kwargs):
        return []

    def get_test_transformer(self, *args, **kwargs):
        return []