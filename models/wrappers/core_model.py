
from models.wrappers.model_interface import ModelInterface
from models.enum_step_types import StepType
from dataset.dataset_interface import EnumDatasetTaskType
from evaluator.average_metric import AverageMeter
from evaluator.evaluator_factory import EvaluatorFactory
from evaluator.evaluator_factory import NullWriter
from dataset.dataset_interface import EnumDatasetTaskType
from models.enum_step_types import StepType

from pathlib import Path
import pathlib
import math
import numpy as np
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import utility.logger as logger

logger_handle = logger.get_logger("vpt_demographic_adaptation")


class CoreModel(ModelInterface):
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
        self._m_learning_rate = (
            self._m_config.SOLVER.LEARNING_RATE * self._m_config.DATASET.BATCH_SIZE / 256
        )
        self._m_linear_keyword = "head"
        
        # Others
        self._m_gpu = in_gpu
        self._m_global_rank = in_global_rank
        self._m_global_world_size = in_global_world_size
        self._m_device = in_device
        self._m_distribution = in_distribution

        # Setup up functions
        self._f_evaluate = self.null_evaluate
        self._m_writer = NullWriter()
        self._f_transmitter_and_receiver = self._transmitter
        self.append_checkpoint = self.null_save
        self.save_checkpoint = self.null_save
        self.save_data = self.null_save

        if self._m_global_rank == 0:
            self._m_writer = SummaryWriter(
                log_dir=pathlib.Path(in_config.TENSORBOARD_DIR)
            )
            self._f_transmitter_and_receiver = self._receiver
            self.save_checkpoint = self.save_model
            self.append_results = self._f_append_results
            self.save_data = self._save_data

        # Setup in parent
        self._m_model = None
        self._m_optimizer = None

    # Setup up functions

    def initialize(self, in_num_of_classes, in_dataset_task_type, **kwargs):
        if self._m_global_rank == 0:
            self._f_evaluate = EvaluatorFactory.get(
                in_dataset_task_type, self._m_writer
            )

    def load_backbone(self, in_model):
        # Load pretrained model before DistributedDataParallel constructor
        backbone_path = Path(self._m_config.MODEL.BACKBONE_PATH)
        if not backbone_path.is_file():
            logger_handle.info(f'ERROR : Backbone path is not valid, "{backbone_path}"')
            exit(-1)
            return None, None

        logger_handle.info(f'STATUS : Loading backbone "{backbone_path}"')
        backbone = None
        if os.path.splitext(backbone_path)[1].lower() == ".npz":
            in_model.load_pretrained(backbone_path)
            backbone = in_model.state_dict()
        else:
            backbone = torch.load(backbone_path, map_location="cpu")

        backbone_state_dict = None
        if "state_dict" in backbone:
            backbone_state_dict = backbone["state_dict"]
        elif "model" in backbone:
            backbone_state_dict = backbone["model"]
        else:
            backbone_state_dict = backbone

        for k in list(backbone_state_dict.keys()):
            # Retain only base_encoder up to before the embedding layer
            if k.startswith("module.base_encoder") and not k.startswith(
                "module.base_encoder.%s" % self._m_linear_keyword
            ):
                # Remove prefix
                backbone_state_dict[k[len("module.base_encoder.") :]] = (
                    backbone_state_dict[k]
                )
                # Delete renamed or unused k
                del backbone_state_dict[k]

        for k in ["head.weight", "head.bias"]:
            if (
                k in backbone_state_dict
                and backbone_state_dict[k].shape != backbone_state_dict[k].shape
            ):
                logger_handle.info(f"Removing key {k} from pretrained checkpoint")
                del backbone_state_dict[k]

        msg = in_model.load_state_dict(backbone_state_dict, strict=False)
        logger_handle.info(f'STATUS : Loaded backbone "{backbone_path}"')
        return msg, backbone_state_dict

    def retrieve_checkpoint(self):
        if not self._m_config.MODEL.CHECKPOINT_PATH:
            logger_handle.info(
                f"STATUS : Checkpoint path is not provided, continuing with default initialization."
            )
            return False

        checkpoint_path = Path(self._m_config.MODEL.CHECKPOINT_PATH)

        if not checkpoint_path.is_file():
            logger_handle.info(
                f'ERROR : Checkpoint path is not valid, "{checkpoint_path}", exiting...'
            )
            exit(-1)

        loc = "cuda:{}".format(self._m_gpu)
        checkpoint = torch.load(checkpoint_path, map_location=loc)
        return checkpoint

    def load_checkpoint(self):
        if not self._m_config.MODEL.CHECKPOINT_PATH:
            logger_handle.info(
                f"STATUS : Checkpoint path is not provided, continuing with default initialization."
            )
            return False

        checkpoint_path = Path(self._m_config.MODEL.CHECKPOINT_PATH)

        if not checkpoint_path.is_file():
            logger_handle.info(
                f'ERROR : Checkpoint path is not valid, "{checkpoint_path}", exiting...'
            )
            exit(-1)

        loc = "cuda:{}".format(self._m_gpu)
        checkpoint = torch.load(checkpoint_path, map_location=loc)

        state_dict = None
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            logger_handle.info(
                f'ERROR : Checkpoint weights could not be retrieved, "{checkpoint_path}"'
            )
            logger_handle.info(f'ERROR : checkpoint = "{checkpoint}"')
            return False

        self._m_model.load_state_dict(state_dict)
        self._m_optimizer.load_state_dict(checkpoint["optimizer"])
        return True

    def is_valid_results_in_checkpoint(self):
        results = self.retrieve_checkpoint()["results"]
        is_valid = "valid" in results
        logger_handle.info(
                    f"STATUS : is_valid_results_in_checkpoint : Incomplete checkpoint found."
                )
        return is_valid

    def is_test_results_in_checkpoint(self):
        results = self.retrieve_checkpoint()["results"]
        is_test = "test" in results
        logger_handle.info(
                    f"STATUS : is_test_results_in_checkpoint : Incomplete checkpoint found."
                )
        return is_test

    def setup_optimizer(self, in_parameters):
        if self._m_config.SOLVER.OPTIMIZER == "adamw":
            self._m_optimizer = torch.optim.AdamW(
                in_parameters,
                self._m_learning_rate,
                weight_decay=self._m_config.SOLVER.WEIGHT_DECAY,
            )
        elif self._m_config.SOLVER.OPTIMIZER == "sgd":
            self._m_optimizer = torch.optim.SGD(
                in_parameters,
                self._m_learning_rate,
                momentum=self._m_config.SOLVER.MOMENTUM,
                weight_decay=self._m_config.SOLVER.WEIGHT_DECAY,
            )
        else:
            logger_handle.info(f'WARNING : optimizer defaulting to SGD"')
            self._m_optimizer = torch.optim.SGD(
                in_parameters,
                self._m_learning_rate,
                momentum=self._m_config.SOLVER.MOMENTUM,
                weight_decay=self._m_config.SOLVER.WEIGHT_DECAY,
            )
        logger_handle.info(f'STATUS : setup_optimizer : Optimizer set up = {self._m_optimizer}"')
        return self._m_optimizer

    def setup_criterion(self, in_task_type: EnumDatasetTaskType):
        if EnumDatasetTaskType.MULTILABEL == in_task_type:
            self._f_criterion = torch.nn.BCEWithLogitsLoss().cuda(self._m_gpu)
        elif EnumDatasetTaskType.MULTICLASS == in_task_type:
            self._f_criterion = torch.nn.CrossEntropyLoss().cuda(self._m_gpu)
        elif EnumDatasetTaskType.BINARY == in_task_type:
            self._f_criterion = torch.nn.BCEWithLogitsLoss().cuda(self._m_gpu)
        else:
            task_str = EnumDatasetTaskType.to_str(in_task_type)
            logger_handle.info(
                f"ERROR : Model is not able to handle the {task_str} task"
            )
            exit(-1)
        logger_handle.info(f'STATUS : setup_criterion : Criterion set up = {self._f_criterion}"')
        

    def distribute(self):
        if self._m_distribution:
            self._m_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._m_model)
            torch.cuda.set_device(self._m_gpu)
            self._m_model.cuda(self._m_gpu)
            self._m_model = torch.nn.parallel.DistributedDataParallel(
                # self._m_model, device_ids=[self._m_gpu]
                self._m_model, device_ids=[self._m_gpu], find_unused_parameters=True
            )
        elif torch.cuda.is_available():
            self._m_model.cuda()
        return self._m_model

    # Step functions

    def backwards(self, in_losses, in_optimizer, in_epoch):
        in_optimizer.zero_grad()
        in_losses.backward()
        in_optimizer.step()

    def train_step(self, in_step_type: StepType, in_data_loader, in_epoch):
        self._m_model.train()

        num_of_batches = len(in_data_loader)

        losses = AverageMeter("Loss", ":.4e")

        epoch_targets: list = []
        epoch_outputs: list = []
        epoch_attribute: list = []

        lr = self._adjust_learning_rate(in_epoch)
        self._m_writer.add_scalar("learning_rate", lr, in_epoch)

        # Run through the epoch.
        for index_batch, (images, targets, attributes) in enumerate(in_data_loader):
            images = images.cuda(self._m_gpu, non_blocking=True)
            targets = targets.cuda(self._m_gpu, non_blocking=True)
            attributes = attributes.cuda(self._m_gpu, non_blocking=True)

            output = self._m_model(images)
            loss = self._f_criterion(output, targets)
            if torch.isnan(loss):
                logger_handle.info(
                    f"STATUS : {in_step_type} [rank {self._m_global_rank}, world size {self._m_global_world_size}, batch {in_epoch}+{index_batch}/{num_of_batches}] : loss = NaN , exiting..."
                )                    
                exit()
            losses.update(loss.item(), images.size(0))

            torch.cuda.synchronize()
            batch_targets, batch_outputs, batch_attributes = (
                self._f_transmitter_and_receiver(targets, output, attributes)
            )

            epoch_targets.extend(batch_targets)
            epoch_outputs.extend(batch_outputs)
            epoch_attribute.extend(batch_attributes)

            if (index_batch + 1) % self._m_config.PRINT_FREQUENCY == 0:
                logger_handle.info(
                    f"STATUS : {in_step_type} [rank {self._m_global_rank}, world size {self._m_global_world_size}, batch {in_epoch}+{index_batch}/{num_of_batches}] : loss = {loss} ({losses.avg})"
                )

            # Compute gradient and do SGD step
            self.backwards(loss, self._m_optimizer, in_epoch)

        self._m_writer.add_scalar(
            in_step_type.to_str() + "EpochLoss", losses.avg, in_epoch
        )

        results = dict()
        results["Loss"] = losses.avg
        if self._m_global_rank > 0:
            return results

        epoch_targets = np.asarray(epoch_targets, dtype=np.float32)
        epoch_outputs = np.asarray(epoch_outputs, dtype=np.float32)

        eval_results = self._f_evaluate(
            in_step_type,
            in_epoch,
            epoch_targets,
            epoch_outputs,
        )
        results.update(eval_results)
        
        logger_handle.info(
            f"Performing {in_step_type} step : Epoch = {in_epoch}, Learning Rate {lr}, Loss {losses.avg}, Results = {results}"
        )
        self._m_writer.flush()
        return results

    def evaluation_step(self, in_step_type: StepType, in_data_loader, in_epoch):
        self._m_model.eval()

        num_of_batches = len(in_data_loader)

        losses = AverageMeter("Loss", ":.5e")

        epoch_targets: list = []
        epoch_outputs: list = []
        epoch_attribute: list = []

        with torch.no_grad():
            for index_batch, (images, targets, attributes) in enumerate(in_data_loader):
                images = images.cuda(self._m_gpu, non_blocking=True)
                targets = targets.cuda(self._m_gpu, non_blocking=True)
                attributes = attributes.cuda(self._m_gpu, non_blocking=True)

                output = self._m_model(images)
                loss = self._f_criterion(output, targets)
                losses.update(loss.item(), images[0].size(0))

                batch_targets, batch_outputs, batch_attributes = (
                    self._f_transmitter_and_receiver(targets, output, attributes)
                )

                epoch_targets.extend(batch_targets)
                epoch_outputs.extend(batch_outputs)
                epoch_attribute.extend(batch_attributes)

                if (index_batch + 1) % self._m_config.PRINT_FREQUENCY == 0:
                    logger_handle.info(
                        f"STATUS : {in_step_type} [rank {self._m_global_rank}, world size {self._m_global_world_size}, batch {in_epoch}+{index_batch}/{num_of_batches}] : loss = {loss}"
                    )

        self._m_writer.add_scalar(
            in_step_type.to_str() + "EpochLoss", losses.avg, in_epoch
        )

        results = dict()
        results["Loss"] = losses.avg
        if self._m_global_rank > 0:
            return results

        epoch_targets = np.asarray(epoch_targets, dtype=np.float32)
        epoch_outputs = np.asarray(epoch_outputs, dtype=np.float32)
        epoch_attribute = np.asarray(epoch_attribute, dtype=np.float32)

        eval_results = self._f_evaluate(
            in_step_type,
            in_epoch,
            epoch_targets,
            epoch_outputs,
        )
        results.update(eval_results)

        if (in_epoch == self._m_config.SOLVER.LAST_EPOCH - 1):
            self.save_data(
                self._m_config.OUTPUT_DIR,
                in_step_type,
                in_epoch,
                epoch_targets,
                epoch_outputs,
                epoch_attribute,
            )

        logger_handle.info(
            f"Performing {in_step_type} step : Epoch = {in_epoch}, Loss {losses.avg}, Results = {results}"
        )
        self._m_writer.flush()
        return results

    # Steps

    def save_model(self, in_start_epoch, in_last_epoch, in_results):
        self._f_save_model(
            in_start_epoch,
            in_last_epoch,
            in_results,
            self._m_config.MODEL.NAME,
            self._m_model,
            self._m_optimizer,
            self._m_config,
            self._m_config.CHECKPOINT_DIR,
        )

    # Helper functions

    def _adjust_learning_rate(self, in_epoch):
        """Decays the learning rate with half-cycle cosine after warmup"""

        lr = self._m_learning_rate

        if in_epoch < self._m_config.SOLVER.WARMUP_EPOCHS:
            lr = self._m_learning_rate * in_epoch / self._m_config.SOLVER.WARMUP_EPOCHS
        else:
            current_lr = self._m_learning_rate
            current_ep = in_epoch - self._m_config.SOLVER.WARMUP_EPOCHS
            total_ep = (
                self._m_config.SOLVER.LAST_EPOCH - self._m_config.SOLVER.WARMUP_EPOCHS
            )
            lr = (
                current_lr * 0.5 * (1.0 + math.cos(math.pi * (current_ep) / (total_ep)))
            )

        for param_group in self._m_optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def _f_save_model(
        self,
        in_start_epoch,
        in_last_epoch,
        in_results,
        in_model_name,
        in_model,
        in_optimizer,
        in_config,
        in_output,
    ):
        # When saving if external storage available create a link to it.
        filepath = pathlib.Path(in_output)
        filename = (
            "ckp_ep_" + str(in_start_epoch) + "_of_" + str(in_last_epoch) + ".pth.tar"
        )
        filepath = filepath.joinpath(filename)

        logger_handle.info(
            f"Saving {in_model_name} model, epoch {in_start_epoch}/{in_last_epoch} to {filepath}"
        )

        torch.save(
            {
                "start_epoch": in_start_epoch,
                "last_epoch": in_last_epoch,
                "results": in_results,
                "config": str(in_config),
                "state_dict": in_model.state_dict(),
                "optimizer": in_optimizer.state_dict(),
            },
            str(filepath),
        )
        
    def _f_append_results(
        self,
        in_start_epoch,
        in_last_epoch,
        in_results,
    ):
        checkpoint = self.retrieve_checkpoint()
        filepath = pathlib.Path(self._m_config.CHECKPOINT_DIR)
        filename = (
            "ckp_ep_" + str(in_start_epoch) + "_of_" + str(in_last_epoch) + ".pth.tar"
        )
        filepath = filepath.joinpath(filename)

        logger_handle.info(
            f"Appending results to {self._m_config.MODEL.NAME} model, epoch {in_start_epoch}/{in_last_epoch} to {filepath}"
        )

        results = checkpoint["results"]
        merged_results = results | in_results
        checkpoint["results"] = merged_results

        torch.save(
            checkpoint,
            str(filepath),
        )

    def _save_data(
        self,
        in_output_dir,
        in_step_type,
        in_epoch,
        in_targets,
        in_outputs,
        in_attribute,
    ):
        filepath_root = pathlib.Path(in_output_dir)
        logger_handle.info(f"Saving targets of shape {in_targets.shape}")
        logger_handle.info(f"Saving output of shape {in_outputs.shape}")
        logger_handle.info(f"Saving attribute of shape {len(in_attribute)}")
        target_filename = (
            in_step_type.to_str() + "_target_" + "epoch_" + str(in_epoch)
        )
        output_filename = (
            in_step_type.to_str() + "_output_" + "epoch_" + str(in_epoch)
        )
        attribute_filename = (
            in_step_type.to_str() + "_attribute_" + "epoch_" + str(in_epoch)
        )
        filepath_target = filepath_root.joinpath(target_filename)
        filepath_output = filepath_root.joinpath(output_filename)
        filepath_attribute = filepath_root.joinpath(attribute_filename)
        np.save(filepath_target, in_targets)
        np.save(filepath_output, in_outputs)
        np.save(filepath_attribute, in_attribute)

    def _transmitter(self, in_target, in_output, in_attributes):
        dist.send(tensor=in_target, dst=0, tag=1)
        dist.send(tensor=in_output, dst=0, tag=2)
        dist.send(tensor=in_attributes, dst=0, tag=3)
        return [], [], []

    def _receiver(self, in_target, in_output, in_attributes):

        targets: list = []
        outputs: list = []
        attributes: list = []

        targets.extend(list(in_target.detach().cpu().numpy()))
        outputs.extend(list(in_output.detach().cpu().numpy()))
        attributes.extend(list(in_attributes.detach().cpu().numpy()))

        for index_rank in range(1, self._m_global_world_size):
            temp_target = torch.empty_like(in_target, device="cuda")
            temp_output = torch.empty_like(in_output, device="cuda")
            temp_attributes = torch.empty_like(in_attributes, device="cuda")

            dist.recv(tensor=temp_target, src=index_rank, tag=1)
            dist.recv(tensor=temp_output, src=index_rank, tag=2)
            dist.recv(tensor=temp_attributes, src=index_rank, tag=3)

            targets.extend(list(temp_target.detach().cpu().numpy()))
            outputs.extend(list(temp_output.detach().cpu().numpy()))
            attributes.extend(list(temp_attributes.detach().cpu().numpy()))

        return targets, outputs, attributes

    def get_train_transformer(self):
        return []

    def get_test_transformer(self):
        return []
