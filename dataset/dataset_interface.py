from enum import Enum
import pandas as pd
from sklearn.model_selection import train_test_split
import utility.logger as logger
import cv2
import torch
import pandas as pd
import pathlib
from torchvision import transforms

import PIL
from torch.utils.data import Dataset

import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


import utility.logger as logger

logger_handle = logger.get_logger("vpt_demographic_adaptation")


class EnumDatasetTaskType(Enum):
    PRETRAINING = 0
    BINARY = 1
    MULTICLASS = 2
    MULTILABEL = 3

    @staticmethod
    def from_str(label):
        if label in ("pretraining"):
            return EnumDatasetTaskType.PRETRAINING
        elif label in ("binary"):
            return EnumDatasetTaskType.BINARY
        elif label in ("multiclass"):
            return EnumDatasetTaskType.MULTICLASS
        elif label in ("multilabel"):
            return EnumDatasetTaskType.MULTILABEL
        else:
            raise NotImplementedError

    def to_str(label):
        if label == EnumDatasetTaskType.PRETRAINING:
            return "pretraining"
        elif label == EnumDatasetTaskType.BINARY:
            return "binary classification"
        elif label == EnumDatasetTaskType.MULTICLASS:
            return "multiclass classification"
        elif label == EnumDatasetTaskType.MULTILABEL:
            return "multilabel classification"
        else:
            raise NotImplementedError


class My_Dataset(Dataset):

    def __init__(
        self,
        in_config,
        in_split="train",
    ):
        super().__init__()
        self._m_config = in_config

        self._f_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    in_config.DATASET.IMAGE_SIZE, interpolation=PIL.Image.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        self._m_split = in_split
        if (in_split == "train") and (in_config.DATASET.TRAINVALID):
            self._m_split = "trainval"

        self._m_pretraining = True
        self._m_train_columns = []
        self._m_images_list = []
        self._m_labels_list = []
        self._m_attribute_list = []
        self._m_class_count = 0
        self._m_num_images = 0

        self._m_cached_images_list = []
        self._m_cached_labels_list = []
        self._m_cached_attribute_list = []
        self.item_getter = self.process_and_retrieve_item

    def log_info(self):
        logger_handle.info(
            f"Member values : My_Dataset.__init__() : "
            f"self._f_transform = {self._f_transform}, "
            f"self._m_pretraining = {self._m_pretraining}, "
            f"self._m_split = {self._m_split}, "
            f"self._m_train_columns = {self._m_train_columns}, "
            f"len(self._m_images_list) = {len(self._m_images_list)}, "
            f"len(self._m_labels_list) = {len(self._m_labels_list)}, "
            f"len(self._m_attribute_list) = {len(self._m_attribute_list)}, "
            f"len(self._m_cached_images_list) = {len(self._m_cached_images_list)}, "
            f"len(self._m_cached_labels_list) = {len(self._m_cached_labels_list)}, "
            f"len(self._m_cached_attribute_list) = {len(self._m_cached_attribute_list)}, "
            f"self._m_class_count = {self._m_class_count}, "
            f"self._m_num_images = {self._m_num_images}"
        )
        logger_handle.info(
            f"Member functions : My_Dataset.__init__() : "
            f"self.task_type = {self.task_type}, "
            f"self.data_size = {self.data_size}, "
            f"self.num_classes = {self.num_classes}, "
            f"self.item_getter = {self.item_getter}, "
            f"self.mean = {self.mean}, "
            f"self.std = {self.std}"
        )

    @property
    def usable(self):
        return True
    
    @property
    def mean(self):
        return (0.485, 0.456, 0.406)

    @property
    def std(self):
        return (0.229, 0.224, 0.225)

    @property
    def task_type(self):
        return EnumDatasetTaskType.PRETRAINING

    @property
    def num_classes(self):
        return self._m_class_count

    @property
    def data_size(self):
        return self._m_num_images

    def __len__(self):
        return self._m_num_images

    def process_and_retrieve_item(self, idx):
        return self.process_items(idx)

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


    def __getitem__(self, idx):
        return self.item_getter(idx)
    
    def analyze_and_log_sensitive_attribute(self, in_df, in_sensitive_column):
        unique_sensitive_attribute_value_counts = in_df[in_sensitive_column].value_counts()
        logger_handle.info(
            f"Sensitive attribute analysis : analyze_and_log_sensitive_attribute() : \n{unique_sensitive_attribute_value_counts}"
        )