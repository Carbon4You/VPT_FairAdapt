from dataset.custom_chest_x_ray import CustomChestXRay
from dataset.chest_x_ray_vtab import ChestXRayVTAB
from dataset.celeba_vtab import CelebA_VTAB
from dataset.celeba import CelebA

LOCAL_DATASETS = {
    "MIMIC_CXR": CustomChestXRay,
    "CheXpert": CustomChestXRay,
    "CelebA": CelebA,
    "ChestXRayVTAB": ChestXRayVTAB,
    "CelebA_VTAB": CelebA_VTAB,
}

import torch
import torch.utils.data

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

class DatasetFactory:
    def __init__(
        self,
        in_config,
        in_dataset_name,
        in_dataset_path,
        in_dataset_data_roots,
        in_dataset_subset=0,
        **kwargs,
    ) -> None:
        self._m_config = in_config
        self._m_dataset_name = in_dataset_name
        self._m_dataset_paths = in_dataset_path
        self._m_dataset_data_paths = in_dataset_data_roots
        self._m_dataset_subset = in_dataset_subset

        self._f_dataset_class_callables = self.callable_dataset_class(
            self._m_dataset_name
        )

    def callable_dataset_class(self, in_dataset_names: str):
        logger_handle.info(f"callable_dataset_class({in_dataset_names})")

        callables = []

        for dataset in in_dataset_names:
            # First check the local datasets
            found_local_dataset = LOCAL_DATASETS[dataset]
            if found_local_dataset:
                callables.append(found_local_dataset)
            
        return callables

    def get_dataset(
        self, in_transformers, in_split="train", in_pretraining=False, **kwargs
    ):

        datasets = []

        num_of_classes = None
        task_type = None

        for index, (callable, path) in enumerate(
            zip(self._f_dataset_class_callables, self._m_dataset_paths)
        ):

            if self._m_dataset_data_paths:
                kwargs["in_dataset_data_roots"] = self._m_dataset_data_paths[index]

            logger_handle.info(f"Callable {callable}, Path {path}")
            dataset = callable(
                self._m_config,
                in_root=path,
                in_split=in_split,
                in_transformers=in_transformers,
                in_pretraining=in_pretraining,
                **kwargs,
            )

            if not dataset.usable:
                continue

            num_of_classes = dataset.num_classes
            task_type = dataset.task_type

            datasets.append(dataset)

        if len(datasets) < 1:
            return None, None, None
        
        datasets = torch.utils.data.ConcatDataset(datasets)

        if (self._m_dataset_subset > 0) & (self._m_dataset_subset < 1):
            ds_size = len(datasets)
            new_size = int(self._m_dataset_subset * ds_size)
            left_over_size = ds_size - new_size

            logger_handle.info(f"STATUS : ds_size = {ds_size}, new_size = {new_size}")

            subset = list(range(1, new_size, 1))
            
            generator = torch.Generator().manual_seed(42)
            datasets, _ = torch.utils.data.random_split(datasets, [new_size, left_over_size], generator=generator)[0]
            
        return datasets, num_of_classes, task_type
