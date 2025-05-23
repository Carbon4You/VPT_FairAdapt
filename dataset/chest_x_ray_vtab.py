import cv2
import torch
import pandas as pd
import pathlib
from torchvision import transforms

import PIL

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

from dataset.dataset_interface import EnumDatasetTaskType
from dataset.dataset_interface import My_Dataset

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")


class ChestXRayVTAB(My_Dataset):
    def __init__(
        self,
        in_config,
        in_root="",
        in_split="train",
        in_dataset_data_roots="",
        in_dataset_data_path_column_name="PATH",
        in_transformers=[],
        in_train_columns=[
            "ENLARGED CARDIOMEDIASTINUM",
            "CARDIOMEGALY",
            "LUNG OPACITY",
            "LUNG LESION",
            "EDEMA",
            "CONSOLIDATION",
            "PNEUMONIA",
            "ATELECTASIS",
            "PNEUMOTHORAX",
            "PLEURAL EFFUSION",
            "PLEURAL OTHER",
            "FRACTURE",
            "SUPPORT DEVICES",
            "NO FINDING",
        ],
        in_pretraining=False,
    ):
        super().__init__(in_config, in_split)
        logger_handle.info(
            f"Entering function : ChestXRayVTAB.__init__("
            f"in_root = {in_root}, "
            f"in_split = {in_split}, "
            f"in_dataset_data_roots = {in_dataset_data_roots}, "
            f"in_dataset_data_path_column_name = {in_dataset_data_path_column_name}, "
            f"in_transformers = {in_transformers}, "
            f"in_train_columns = {in_train_columns})"
        )

        data_root = pathlib.Path(in_root)
        dataset_path = data_root.joinpath(in_config.DATASET.SUBSET)
        split_path = dataset_path.joinpath(self._m_split + ".csv")
        df = pd.read_csv(split_path)

        self._m_pretraining = in_pretraining
        self._m_images_list = df[in_dataset_data_path_column_name].tolist()
        self._m_labels_list = df[in_train_columns].values.tolist()
        self._m_attribute_list = df["RACE_ATTRIBUTE_ID"].tolist()
        self._m_class_count = len(in_train_columns)
        self._m_num_images = len(df)

        self.analyze_and_log_sensitive_attribute(df, "RACE")
        self.log_info()

    @property
    def task_type(self):
        return EnumDatasetTaskType.MULTILABEL

    @property
    def mean(self):
        return (0.483501, 0.483501, 0.483501)

    @property
    def std(self):
        return (0.300156, 0.300156, 0.300156)

    def process_items(self, idx):
        gray_image = cv2.imread(self._m_images_list[idx], cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            logger_handle.info(f"ERROR : Issues with gray_image file {self._m_images_list[idx]}")
            
        bgr_image =None
        try:
            bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        except:
            logger_handle.info(f"ERROR : Issues with cvtColor file {self._m_images_list[idx]}")
            print(f"ERROR : Issues with cvtColor file {self._m_images_list[idx]}")
        
        image = self._f_transform(bgr_image)

        label = torch.tensor(self._m_labels_list[idx], dtype=torch.float32).reshape(-1)
        if self._m_pretraining:
            label = -1

        attribute = torch.tensor(
            self._m_attribute_list[idx], dtype=torch.float32
        ).reshape(-1)
        return image, label, attribute
