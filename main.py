from dataset.dataset_factory import DatasetFactory
from models.model_factory import ModelFactory
from models.enum_step_types import StepType
from utility.config import get_cfg
from utility.config_node import PathManager
import utility.logger as logger

import argparse
import builtins
import os
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import random
import pathlib
import glob
import numpy as np

def default_argument_parser():
    """
    create a simple parser to wrap around config file
    """
    parser = argparse.ArgumentParser(description="visual-prompt")
    parser.add_argument(
        "--config_file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def parameter_folder_name(in_config):
    name = str(in_config.SOLVER.OPTIMIZER)
    name += "_bs" + str(in_config.DATASET.BATCH_SIZE)
    name += "_lr" + str(in_config.SOLVER.LEARNING_RATE)
    name += "_wd" + str(in_config.SOLVER.WEIGHT_DECAY)
    if "E2VPT" in str(in_config.MODEL.NAME):
        name += "_e2pr" + str(in_config.MODEL.E2VPT.KV_PROMPT.NUM_TOKENS)
        name += "_e2prp" + str(in_config.MODEL.E2VPT.KV_PROMPT.NUM_TOKENS_P)
    elif "VPT" in str(in_config.MODEL.NAME):
        name += "_pr" + str(in_config.MODEL.PROMPT.NUM_TOKENS)

    if "gate" in str(in_config.MODEL.ARCH):
        name += "_gpr" + str(in_config.MODEL.GATED.PROMPT.NUM_TOKENS)
        name += "_gini" + str(in_config.MODEL.GATED.PROMPT.GATE_INIT)
    name += "_wu" + str(in_config.SOLVER.WARMUP_EPOCHS)
    name += "_le" + str(in_config.SOLVER.LAST_EPOCH)
    return name


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Path : project_root/MODEL/ARCH/TRANSFER/DATASET/SUBSET/working_directory/checkpoints/
    # Path : project_root/MODEL/ARCH/TRANSFER/DATASET/SUBSET/working_directory/tensorboard/
    # Path : project_root/MODEL/ARCH/TRANSFER/DATASET/SUBSET/working_directory/logs/

    working_directory = parameter_folder_name(cfg)
    dataset_name = "-".join(cfg.DATASET.NAMES)

    output_folder = os.path.join(
        cfg.OUTPUT_DIR,
        cfg.MODEL.NAME,
        cfg.MODEL.ARCH,
        cfg.MODEL.PRETRAINED_DATASET,
        dataset_name,
        cfg.DATASET.SUBSET,
        working_directory,
    )

    checkpoints_folder = os.path.join(output_folder, "checkpoints")
    tensorboard_folder = os.path.join(output_folder, "tensorboard")
    logs_folder = os.path.join(output_folder, "logs")

    cfg.OUTPUT_DIR = output_folder
    cfg.CHECKPOINT_DIR = checkpoints_folder
    cfg.TENSORBOARD_DIR = tensorboard_folder
    cfg.LOG_DIR = logs_folder

    if cfg.COMMAND_CHECK_IF_COMPLETED:
        return cfg

    if not PathManager.exists(output_folder):
        PathManager.mkdirs(checkpoints_folder)
        PathManager.mkdirs(tensorboard_folder)
        PathManager.mkdirs(logs_folder)

    pattern = "**/*.tar"
    matches = glob.glob(
        pattern,
        root_dir=pathlib.Path(checkpoints_folder),
        recursive=True,
        include_hidden=True,
    )
    for file in matches:
        print(file)

    # Search for resume checkpoint
    if matches:
        checkpoint_epochs = list()
        for checkpoint in matches:
            checkpoint_epochs.append(int(checkpoint.split("_")[2]))

        print(f"INFO: Checkpoint epochs = {checkpoint_epochs}")

        max_epoch_index = np.argmax(checkpoint_epochs)
        checkpoint_file = matches[max_epoch_index]
        resume_epoch = checkpoint_epochs[max_epoch_index]

        print(f"INFO: checkpoint_file = {checkpoint_file}")
        print(f"INFO: resume_epoch = {resume_epoch}")
        cfg.SOLVER.FIRST_EPOCH = int(resume_epoch) + 1
        cfg.MODEL.CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_DIR, checkpoint_file)

    # Create a link to checkpoint directory if required.
    if cfg.CHECKPOINT_LINKING:
        cfg.SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]

        symlink_original_folder = cfg.CHECKPOINT_ROOT + cfg.SLURM_JOB_ID
        print(f"INFO: symlink_original_folder = {symlink_original_folder}")
        if not PathManager.exists(symlink_original_folder):
            PathManager.mkdirs(symlink_original_folder)

        cfg.CHECKPOINT_DIR = os.path.join(cfg.CHECKPOINT_DIR, cfg.SLURM_JOB_ID)
        if cfg.RANK == 0:
            os.symlink(symlink_original_folder, cfg.CHECKPOINT_DIR)

    print(f"INFO: CHECKPOINT_DIR = {cfg.CHECKPOINT_DIR}")
    print(f"INFO: TENSORBOARD_DIR = {cfg.TENSORBOARD_DIR}")
    print(f"INFO: LOG_DIR = {cfg.LOG_DIR}")

    cfg.freeze()
    return cfg


def print_config(in_config):
    print("\n\nConfig")
    print("----------------------------------------------------------------------")
    print("Configuration Start")
    print("----------------------------------------------------------------------")
    print(in_config)
    print("----------------------------------------------------------------------")
    print("Configuration End")
    print("----------------------------------------------------------------------")
    print("\n\n")


class Application:

    def torch_worker(self, in_gpu: int, ngpus_per_node: int, in_config: dict):
        global_world_size = ngpus_per_node * in_config.WORLD_SIZE
        global_rank = in_config.RANK * ngpus_per_node + in_gpu
        logger_handle = logger.setup_logging(
            global_rank, global_world_size, in_config.LOG_DIR, name="vpt_demographic_adaptation"
        )
        logger_handle.info(
            f"INFO: torch_worker : Starting at gpu = {in_gpu}, global_world_size = {str(global_world_size)}, global_rank = {str(global_rank)}"
        )
        logger_handle.info(f"INFO: torch_worker : Configuration {str(in_config)}")

        distributed = in_config.WORLD_SIZE > 1 or in_config.MULTIPROCESSING_DISTRIBUTED
        logger_handle.info(f"INFO: torch_worker : Distributed {distributed}")

        if distributed:
            if in_gpu != 0:

                def null_print(*args, flush=True):
                    pass

                builtins.print = null_print

        if in_config.CUDA:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
        logger_handle.info(f"INFO: torch_worker : Device is {device}")

        batch_size = int(in_config.DATASET.BATCH_SIZE)

        if distributed:

            batch_size = int(batch_size / global_world_size)
            logger_handle.info(
                f"INFO: torch_worker : New batch size per GPU is {batch_size}"
            )

            dist.init_process_group(
                backend=in_config.DIST_BACKEND,
                init_method=in_config.DIST_URL,
                world_size=global_world_size,
                rank=global_rank,
            )
            torch.distributed.barrier()

        model_wrapper = ModelFactory(in_config.MODEL.NAME).get_model_wrapper(
            in_config=in_config,
            in_distribution=distributed,
            in_device=device,
            in_gpu=in_gpu,
            in_global_rank=global_rank,
            in_global_world_size=global_world_size,
        )

        dataset_factory = DatasetFactory(
            in_config,
            in_config.DATASET.NAMES,
            in_config.DATASET.PATHS,
            in_config.DATASET.DATA_PATHS,
        )

        train_dataset, num_of_classes, task_type = dataset_factory.get_dataset(
            model_wrapper.get_train_transformer(), "train", False
        )

        validation_dataset = None
        test_dataset = None
        if in_config.DATASET.VALIDATE:
            validation_dataset, _, _ = dataset_factory.get_dataset(
                model_wrapper.get_test_transformer(), "valid"
            )
        if in_config.DATASET.TEST:
            test_dataset, _, _ = dataset_factory.get_dataset(
                model_wrapper.get_test_transformer(), "test"
            )

        train_sampler = None
        validation_sampler = None
        test_sampler = None
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True
            )
            if validation_dataset:
                validation_sampler = torch.utils.data.distributed.DistributedSampler(
                    validation_dataset, shuffle=False
                )
            if test_dataset:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_dataset, shuffle=False
                )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=in_config.DATASET.WORKERS,
            pin_memory=True,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            persistent_workers=True,
        )

        if validation_dataset:
            validation_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=in_config.DATASET.WORKERS,
                sampler=validation_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
        if test_dataset:
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=in_config.DATASET.WORKERS,
                sampler=test_sampler,
            )

        model_wrapper.initialize(
            num_of_classes, task_type, in_train_loader=train_loader
        )
        if (in_config.SOLVER.FIRST_EPOCH == 0) and (model_wrapper.model() != None):
            logger_handle.info(
                f"INFO: torch_worker : Initialized model is {model_wrapper.model()}"
            )
            logger_handle.info(
                f"INFO: torch_worker : Initialized model parameters are {dict(model_wrapper.model().named_parameters())}"
            )
            logger_handle.info(
                f"INFO: torch_worker : Trainable model parameters are :"
            )
            for parameter, value in model_wrapper.model().named_parameters():
                if value.requires_grad:
                    logger_handle.info(f"\t {parameter}")
            trainable_parameters = filter(
                lambda p: p.requires_grad, model_wrapper.model().parameters()
            )
            all_parameters = model_wrapper.model().parameters()
            trainable_parameters = sum(
                [np.prod(p.size()) for p in trainable_parameters]
            )
            all_parameters = sum([np.prod(p.size()) for p in all_parameters])
            percentage_trainable = trainable_parameters / all_parameters
            logger_handle.info(
                f"INFO: torch_worker : Initialized model, total parameters: {all_parameters} , trainable parameters: {trainable_parameters} , percentage trainable {percentage_trainable}"
            )

        # cudnn.benchmark = True

        if in_config.EVALUATE == "validation":
            model_wrapper.evaluation_step(
                StepType.VALIDATING, validation_loader, in_config.SOLVER.FIRST_EPOCH
            )
            return
        elif in_config.EVALUATE == "test":
            model_wrapper.evaluation_step(
                StepType.TESTING, test_loader, in_config.SOLVER.FIRST_EPOCH
            )
            return

        for epoch in range(in_config.SOLVER.FIRST_EPOCH, in_config.SOLVER.LAST_EPOCH):
            if distributed:
                train_loader.sampler.set_epoch(epoch)
                if validation_dataset:
                    validation_loader.sampler.set_epoch(epoch)
                if test_dataset:
                    test_loader.sampler.set_epoch(epoch)

            logger_handle.info(
                f"INFO: torch_worker : Epoch {epoch} started of {in_config.SOLVER.LAST_EPOCH}"
            )

            train_results = model_wrapper.train_step(
                StepType.TRAINING, train_loader, epoch
            )
            torch.cuda.synchronize()

            valid_results = None
            if (validation_dataset) and (
                (epoch % in_config.DATASET.VALIDATION_FREQUENCY == 0)
                or (epoch == in_config.SOLVER.LAST_EPOCH - 1)
            ):
                valid_results = model_wrapper.evaluation_step(
                    StepType.VALIDATING, validation_loader, epoch
                )
                torch.cuda.synchronize()

            test_results = None
            if test_dataset and (
                (epoch % in_config.DATASET.TEST_FREQUENCY == 0)
                or (epoch == in_config.SOLVER.LAST_EPOCH - 1)
            ):
                test_results = model_wrapper.evaluation_step(
                    StepType.TESTING, test_loader, epoch
                )
                torch.cuda.synchronize()

            if (
                (epoch % in_config.MODEL.SAVE_FREQUENCY == 0)
                or (epoch == in_config.SOLVER.LAST_EPOCH - 1)
            ) and (in_config.MODEL.SAVE_CHECKPOINT):
                model_wrapper.save_checkpoint(
                    epoch,
                    in_config.SOLVER.LAST_EPOCH,
                    {
                        "train": train_results,
                        "valid": valid_results,
                        "test": test_results,
                    },
                )

    def main(self):
        print("INFO: Experiment Started ...")

        config = setup(default_argument_parser().parse_args())
        print_config(config)

        if config.SEED is not None:
            random.seed(config.SEED)
            torch.manual_seed(config.SEED)
            cudnn.deterministic = True

        if config.COMMAND_CHECK_IF_COMPLETED:
            test_output_file = pathlib.Path(config.OUTPUT_DIR).joinpath(
                "TESTING_output_epoch_" + str(config.SOLVER.LAST_EPOCH - 1) + ".npy"
            )
            test_output_file_exists = test_output_file.exists()
            valid_output_file = pathlib.Path(config.OUTPUT_DIR).joinpath(
                "VALIDATING_output_epoch_" + str(config.SOLVER.LAST_EPOCH - 1) + ".npy"
            )
            valid_output_file_exists = valid_output_file.exists()
            if test_output_file_exists or valid_output_file_exists:
                exit(58)
            else:
                exit(404)

        has_cuda = torch.cuda.is_available()
        if not has_cuda:
            self.torch_worker(0, 0, config)
        else:
            ngpus_per_node = torch.cuda.device_count()
            mp.spawn(
                self.torch_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config)
            )

        print("Experiment Finished ...")
        return 0


if __name__ == "__main__":
    app = Application()
    app.main()
