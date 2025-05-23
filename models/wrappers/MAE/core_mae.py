from models.wrappers.core_model import CoreModel
from dataset.dataset_interface import EnumDatasetTaskType
import models.wrappers.MAE.mae_vit
import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

import torch

class CoreMAE(CoreModel):
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

    def setup_model(self, in_num_of_classes):
        model = models.wrappers.MAE.mae_vit.__dict__[self._m_config.MODEL.ARCH](
            in_config=self._m_config,
            num_classes=in_num_of_classes,
            img_size=self._m_config.DATASET.IMAGE_SIZE,
            drop_rate=self._m_config.MODEL.DROP_RATE,
            drop_path_rate=self._m_config.MODEL.DROP_RATE_PATH,
        )
        return model

    def interpolate_pos_embed(self, model, checkpoint_model):
        if "pos_embed" in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model["pos_embed"]
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches**0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                logger_handle.info(
                    "Position interpolate from %dx%d to %dx%d"
                    % (orig_size, orig_size, new_size, new_size)
                )
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(
                    -1, orig_size, orig_size, embedding_size
                ).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model["pos_embed"] = new_pos_embed
