from models.wrappers.ViT.e2vpt_vit import E2VPT_ViT
from models.wrappers.ViT.vpt_vit import VPT_ViT
from models.wrappers.ViT.full_vit import FullViT
from models.wrappers.ViT.linear_vit import LinearViT

from models.wrappers.MoCoV3.e2vpt_mocov3 import E2VPT_MoCoV3
from models.wrappers.MoCoV3.vpt_mocov3 import PromptedMoCoV3
from models.wrappers.MoCoV3.full_mocov3 import FullMoCoV3
from models.wrappers.MoCoV3.linear_mocov3 import LinearMoCoV3

from models.wrappers.MAE.e2vpt_mae import E2VPT_MAE
from models.wrappers.MAE.vpt_mae import PromptedMAE
from models.wrappers.MAE.full_mae import FullMAE
from models.wrappers.MAE.linear_mae import LinearMAE

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

class ModelFactory:
    def __init__(self, in_model_name: str, **kwargs) -> None:
        self._m_model_name = in_model_name

        self._f_model_class_callable = self.callable_model_class(self._m_model_name)

    def callable_model_class(self, in_model_name: str):

        # First check the local datasets
        found_local_model = {
            # ViT
            "E2VPT_ViT": E2VPT_ViT,
            "VPT_ViT": VPT_ViT,
            "GVPT_ViT": VPT_ViT,
            "Full_ViT": FullViT,
            "Linear_ViT": LinearViT,
            # MAE
            "E2VPT_MAE": E2VPT_MAE,
            "VPT_MAE": PromptedMAE,
            "GVPT_MAE": PromptedMAE,
            "Full_MAE": FullMAE,
            "Linear_MAE": LinearMAE,
            # MoCoV3
            "E2VPT_MoCoV3": E2VPT_MoCoV3,
            "VPT_MoCoV3": PromptedMoCoV3,
            "GVPT_MoCoV3": PromptedMoCoV3,
            "Full_MoCoV3": FullMoCoV3,
            "Linear_MoCoV3": LinearMoCoV3,
        }

        if found_local_model:
            return found_local_model[in_model_name]
        return None

    def get_model_wrapper(self, **kwargs):
        model = self._f_model_class_callable(**kwargs)
        return model
