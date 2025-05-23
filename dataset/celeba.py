import torch
import pandas as pd
import pathlib
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True
from timm.data import create_transform
from dataset.dataset_interface import EnumDatasetTaskType

from sklearn.model_selection import train_test_split
import numpy as np
from dataset.dataset_interface import My_Dataset
import utility.logger as logger

logger_handle = logger.get_logger("vpt_demographic_adaptation")


class CelebA(My_Dataset):
    def __init__(
        self,
        in_config,
        in_root="",
        in_split="train",
        in_dataset_data_roots="",
        in_dataset_data_path_column_name="PATH",
        in_transformers=[],
        in_train_columns=[
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Gray_Hair",
            "High_Cheekbones",
            "Mouth_Slightly_Open",
            "Narrow_Eyes",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Necklace",
            "Young",
        ],
        in_pretraining=False,
    ):
        super().__init__(in_config, in_split)
        logger_handle.info(
            f"Entering function : CelebA.__init__("
            f"in_root = {in_root}, "
            f"in_split = {in_split}, "
            f"in_dataset_data_roots = {in_dataset_data_roots}, "
            f"in_dataset_data_path_column_name = {in_dataset_data_path_column_name}, "
            f"in_transformers = {in_transformers}, "
            f"in_train_columns = {in_train_columns})"
        )
        
        split_id = 0
        if self._m_split == "valid":
            split_id = 1
        elif self._m_split == "test":
            split_id = 2

        in_root = pathlib.Path(in_root)
        split_path = in_root.joinpath("list_eval_partition.txt")
        label_path = in_root.joinpath("list_attr_celeba.txt")

        split_df = pd.read_csv(
            split_path,
            delim_whitespace=True,
            names=[in_dataset_data_path_column_name, "split"],
        )
        label_df = pd.read_csv(
            label_path, skiprows=[0], delim_whitespace=True, header=0
        ).reset_index()
        label_df.rename(
            columns={"index": in_dataset_data_path_column_name}, inplace=True
        )

        df = pd.merge(
            split_df, label_df, on=in_dataset_data_path_column_name, how="inner"
        )
        df[in_dataset_data_path_column_name] = (
            in_dataset_data_roots + df[in_dataset_data_path_column_name]
        )

        df = df.replace(-1, 0)

        df = df.loc[df["split"] == split_id]

        seed=1234567
        if in_config.DATASET.SUBSET == "FEMALE":
            df = df.loc[df["Male"] == 0]
            seed=7
        elif in_config.DATASET.SUBSET == "MALE":
            df = df.loc[df["Male"] == 1]

        self._m_pretraining = in_pretraining
        self._m_attribute_list = df["Male"].tolist()
        self._m_train_columns = in_train_columns
        self._m_images_list = df[in_dataset_data_path_column_name].tolist()
        self._m_labels_list = df[self._m_train_columns].values.tolist()
        self._m_class_count = len(self._m_train_columns)
        self._m_num_images = len(self._m_images_list)
        
        self.analyze_and_log_sensitive_attribute(df, "Male")
        self.log_info()
        
    @property
    def task_type(self):
        return EnumDatasetTaskType.MULTILABEL

    def subset_stratify(
        self,
        in_df,
        in_subset,
        in_subset_amount,
        in_sensitive_column,
        in_train_columns,
    ):
        in_subset_amount = int(in_subset_amount)
        # stratify_columns = in_train_columns.copy()
        subset_df = in_df.copy()
        
        # if in_subset == "ALL":
            # stratify_columns.append(in_sensitive_column)
        
        # subset_df["stratify"] = subset_df[stratify_columns].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
        
        # unique_df: pd.DataFrame = subset_df.loc[subset_df.groupby("stratify").filter(lambda x: len(x) == 1).index]
        
        # if len(unique_df) > in_subset_amount:
        #     return unique_df.sample(n=in_subset_amount, random_state=0)
        
        # subset_df = subset_df.loc[subset_df.groupby("stratify").filter(lambda x: len(x) > 1).index]
        # in_subset_amount -= len(unique_df)
        
        train_subset_df, _ = train_test_split(
            subset_df,
            train_size=in_subset_amount,
            test_size=None,
            random_state=7,
            stratify=in_df[in_sensitive_column]
        )
        # subset_df = train_subset_df.copy()
        # subset_df = pd.concat([train_subset_df, unique_df], ignore_index=True, axis=0)
        
        # print(f"subset_df = {subset_df}")
        
        return train_subset_df



class PytorchCelebA(torchvision.datasets.celeba.CelebA):
    def __init__(
        self,
        in_config,
        in_root="",
        in_split="train",
        in_dataset_data_roots="",
        in_dataset_data_path_column_name="",
        in_transformers=[],
        in_train_columns=[],
        in_pretraining=False,
        in_exclude_attribute={},
    ):

        in_root = pathlib.Path(in_root)

        logger_handle.info(f"in_root = {in_root}")
        logger_handle.info(f"in_split = {in_split}")
        logger_handle.info(f"in_dataset_data_roots = {in_dataset_data_roots}")
        logger_handle.info(
            f"in_dataset_data_path_column_name = {in_dataset_data_path_column_name}"
        )
        logger_handle.info(f"in_train_columns = {in_train_columns}")
        logger_handle.info(f"in_transformers = {in_transformers}")

        transformers = []
        transformers.append(
            transforms.Resize(
                in_config.DATASET.IMAGE_SIZE, interpolation=PIL.Image.BICUBIC
            )
        )
        transformers.append(transforms.ToTensor())
        transformers.append(torchvision.transforms.Normalize(self.mean, self.std))
        self._f_transform = torchvision.transforms.Compose(transformers)
        
        super().__init__(
            root=in_dataset_data_roots,
            split=in_split,
            target_type="attr",
            transform=self._f_transform,
            download=True,
        )
        
        self._m_class_count = len(self.attr_names)
        
        logger_handle.info(
            f"STATUS : PytorchCelebA.__init__(...) : self.attr_names = {self.attr_names}"
        )
        
        logger_handle.info(
            f"STATUS : PytorchCelebA.__init__(...) : self._f_transform = {self._f_transform}"
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
        return EnumDatasetTaskType.MULTILABEL

    @property
    def num_classes(self):
        return self._m_class_count

    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)
        return x, y, -1

