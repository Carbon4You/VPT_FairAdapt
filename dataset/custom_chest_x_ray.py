import pandas as pd
import pathlib
from dataset.dataset_interface import EnumDatasetTaskType
from dataset.dataset_interface import My_Dataset
import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")


class CustomChestXRay(My_Dataset):
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

        filename = str(in_split) + ".csv"
        df_path = pathlib.Path(in_root).joinpath(filename)
        df = pd.read_csv(df_path)

        df = df.loc[df["CUSTOM_SPLIT"] == in_split.upper()]
        df[in_dataset_data_path_column_name] = (
            in_dataset_data_roots + df[in_dataset_data_path_column_name]
        )

        def attribute(x):
            if x.startswith("WHITE"):
                return 1
            elif x.startswith("BLACK"):
                return 2
            elif x.startswith("ASIAN"):
                return 3
            return 0
        df["RACE_ATTRIBUTE_ID"] = df["RACE"].map(attribute)
        
        self._m_images_list = df[in_dataset_data_path_column_name].tolist()
        self._m_labels_list = df[in_train_columns].values.tolist()
        self._m_attribute_list = df["RACE_ATTRIBUTE_ID"].tolist()
        self._m_pretraining = in_pretraining
        self._m_split = in_split
        self._m_class_count = len(in_train_columns)
        self._m_num_images = len(df)

        self._m_cached_images_list = []
        self._m_cached_labels_list = []
        self._m_cached_attribute_list = []

        self.analyze_and_log_sensitive_attribute(df, "RACE")
        self.log_info()

    @property
    def usable(self):
        return True

    @property
    def task_type(self):
        return EnumDatasetTaskType.MULTILABEL

    @property
    def num_classes(self):
        return self._m_class_count

    @property
    def mean(self):
        return (0.5056, 0.5056, 0.5056)

    @property
    def std(self):
        return (0.252, 0.252, 0.252)

