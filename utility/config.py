#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from utility.config_node import CfgNode

# Global config object
_GLOBAL_CONFIG_FILE = CfgNode()
#
#
# ----------------------------------------------------------------------
# General options
# ----------------------------------------------------------------------
#
#
_GLOBAL_CONFIG_FILE.COMMAND_CHECK_IF_COMPLETED = False
_GLOBAL_CONFIG_FILE.OUTPUT_DIR = "./data/output/"
_GLOBAL_CONFIG_FILE.LOG_DIR = ""
_GLOBAL_CONFIG_FILE.CHECKPOINT_ROOT = "./data/checkpoints/"
_GLOBAL_CONFIG_FILE.CHECKPOINT_DIR = ""
_GLOBAL_CONFIG_FILE.CHECKPOINT_LINKING = False
_GLOBAL_CONFIG_FILE.TENSORBOARD_DIR = ""
_GLOBAL_CONFIG_FILE.WORKING_ENVIRONMENT = "slurm"
_GLOBAL_CONFIG_FILE.CUDNN_BENCHMARK = False
_GLOBAL_CONFIG_FILE.SEED = None
_GLOBAL_CONFIG_FILE.DIST_BACKEND = "nccl"
_GLOBAL_CONFIG_FILE.DIST_URL = "env://"
_GLOBAL_CONFIG_FILE.DIST_INIT_FILE = ""
_GLOBAL_CONFIG_FILE.RANK = 0
_GLOBAL_CONFIG_FILE.WORLD_SIZE = 1
_GLOBAL_CONFIG_FILE.MULTIPROCESSING_DISTRIBUTED = True
_GLOBAL_CONFIG_FILE.CUDA = True
_GLOBAL_CONFIG_FILE.SLURM_JOB_ID = None
_GLOBAL_CONFIG_FILE.EVALUATE = False
_GLOBAL_CONFIG_FILE.PRINT_FREQUENCY = 100
#
#
# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
#
#
_GLOBAL_CONFIG_FILE.MODEL = CfgNode()
_GLOBAL_CONFIG_FILE.MODEL.NAME = "NullModel"
_GLOBAL_CONFIG_FILE.MODEL.ARCH = "vit_base"
_GLOBAL_CONFIG_FILE.MODEL.EVALUATE = False
_GLOBAL_CONFIG_FILE.MODEL.PRETRAINED_DATASET = "ImageNet"  
_GLOBAL_CONFIG_FILE.MODEL.BACKBONE_PATH = ""  
_GLOBAL_CONFIG_FILE.MODEL.CHECKPOINT_PATH = ""  
_GLOBAL_CONFIG_FILE.MODEL.SAVE_CHECKPOINT = True
_GLOBAL_CONFIG_FILE.MODEL.SAVE_FREQUENCY = 1
_GLOBAL_CONFIG_FILE.MODEL.MODEL_ROOT = ""  # root folder for pretrained model weights
_GLOBAL_CONFIG_FILE.MODEL.MLP_NUM = 1
_GLOBAL_CONFIG_FILE.MODEL.NUM_LAYERS = 12
_GLOBAL_CONFIG_FILE.MODEL.DROP_RATE = 0.1
_GLOBAL_CONFIG_FILE.MODEL.DROP_RATE_PATH = 0.1
# _GLOBAL_CONFIG_FILE.MODEL.LAYER_DECAY = 0.75
# _GLOBAL_CONFIG_FILE.MODEL.POS_DROP_RATE = 0.1
# _GLOBAL_CONFIG_FILE.MODEL.PATCH_DROP_RATE = 0.1
# _GLOBAL_CONFIG_FILE.MODEL.PROJ_DROP_RATE = 0.1
# _GLOBAL_CONFIG_FILE.MODEL.ATTN_DROP_RATE = 0.1
#
#
# ----------------------------------------------------------------------
# Full options
# ----------------------------------------------------------------------
#
#
_GLOBAL_CONFIG_FILE.MODEL.FULL = CfgNode()
#
#
# ----------------------------------------------------------------------
# Linear options
# ----------------------------------------------------------------------
#
#
_GLOBAL_CONFIG_FILE.MODEL.LINEAR = CfgNode()
_GLOBAL_CONFIG_FILE.MODEL.LINEAR.MLP_SIZES = []
_GLOBAL_CONFIG_FILE.MODEL.LINEAR.DROPOUT = 0.1
#
#
# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
#
#
_GLOBAL_CONFIG_FILE.MODEL.PROMPT = CfgNode()
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.NUM_TOKENS = 5
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.DISTANCE = 30
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.LOCATION = "prepend"
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.INITIATION = "random"  # "final-cls", "cls-first12"
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.DEEP = True # "whether do deep prompt or not, only for prepend location"
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt
# how to get the output emb for cls head:
    # original: follow the orignial backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.VIT_POOL_TYPE = "original"
_GLOBAL_CONFIG_FILE.MODEL.PROMPT.DROPOUT = 0.1
#
#
# ----------------------------------------------------------------------
# Gated VPT options
# ----------------------------------------------------------------------
#
#
_GLOBAL_CONFIG_FILE.MODEL.GATED = CfgNode()
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT = CfgNode()
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.NUM_TOKENS = 100
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.LOCATION = "prepend"
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.DROPOUT = 0.1
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.TEMP = 1.0
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.TEMP_LEARN = True
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.INITIATION = "random"
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.GATE_PRIOR = True
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.GATE_NUM = 11
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.GATE_INIT = 10
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.TEMP_NUM = 12
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.TEMP_MIN = 0.01
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.TEMP_MAX = 10.0
_GLOBAL_CONFIG_FILE.MODEL.GATED.PROMPT.VIT_POOL_TYPE = "original" 

#
#
# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
#
#
_GLOBAL_CONFIG_FILE.SOLVER = CfgNode()
_GLOBAL_CONFIG_FILE.SOLVER.OPTIMIZER = "sgd"
_GLOBAL_CONFIG_FILE.SOLVER.LEARNING_RATE = 0.1
_GLOBAL_CONFIG_FILE.SOLVER.WEIGHT_DECAY = 0.0001
_GLOBAL_CONFIG_FILE.SOLVER.SCHEDULER = "cosine"
_GLOBAL_CONFIG_FILE.SOLVER.MOMENTUM = 0.9
_GLOBAL_CONFIG_FILE.SOLVER.WEIGHT_DECAY_BIAS = 0
_GLOBAL_CONFIG_FILE.SOLVER.WARMUP_EPOCHS = 5
_GLOBAL_CONFIG_FILE.SOLVER.FIRST_EPOCH = 0
_GLOBAL_CONFIG_FILE.SOLVER.LAST_EPOCH = 100
_GLOBAL_CONFIG_FILE.SOLVER.LOG_EVERY_N = 100
#
#
# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
#
#
_GLOBAL_CONFIG_FILE.DATASET = CfgNode()
_GLOBAL_CONFIG_FILE.DATASET.NAMES = []
_GLOBAL_CONFIG_FILE.DATASET.SUBSET = ""
_GLOBAL_CONFIG_FILE.DATASET.PATHS = []
_GLOBAL_CONFIG_FILE.DATASET.DATA_PATHS = []
_GLOBAL_CONFIG_FILE.DATASET.SHUFFLE = False
_GLOBAL_CONFIG_FILE.DATASET.IMAGE_SIZE = [224, 224]
_GLOBAL_CONFIG_FILE.DATASET.BATCH_SIZE = 128
_GLOBAL_CONFIG_FILE.DATASET.WORKERS = 8
_GLOBAL_CONFIG_FILE.DATASET.NUM_CHANNELS = 3
_GLOBAL_CONFIG_FILE.DATASET.PIN_MEMORY = True
# TRAINING
_GLOBAL_CONFIG_FILE.DATASET.TRAINVALID = False
# VALIDATION
_GLOBAL_CONFIG_FILE.DATASET.VALIDATE = True
_GLOBAL_CONFIG_FILE.DATASET.VALIDATION_FREQUENCY = 99
# TESTING
_GLOBAL_CONFIG_FILE.DATASET.TEST = True
_GLOBAL_CONFIG_FILE.DATASET.VALIDTEST = False
_GLOBAL_CONFIG_FILE.DATASET.TEST_FREQUENCY = 99
#
#
# ----------------------------------------------------------------------
# E2VPT.KV_PROMPT (Prompt with key and value) options
# ----------------------------------------------------------------------
_GLOBAL_CONFIG_FILE.MODEL.E2VPT = CfgNode()
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT = CfgNode()
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.NUM_TOKENS_P = 5
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.SEARCH_NUM_TOKENS_P = [5, 10, 25, 50, 100, 150, 200]
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.LOCATION = "prepend"
# prompt initalizatioin: 
    # (1) default "random"
    # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
    # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
    # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.INITIATION = "random"  # "final-cls", "cls-first12"
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.CLSEMB_FOLDER = ""
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.CLSEMB_PATH = ""
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.PROJECT = -1  # "projection mlp hidden dim"
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.DEEP_P = True # "whether do deep prompt or not, only for prepend location"


_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt
# how to get the output emb for cls head:
    # original: follow the orignial backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.VIT_POOL_TYPE = "original"
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.DROPOUT_P = 0.1
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.SAVE_FOR_EACH_EPOCH = False

_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.NUM_TOKENS = 5
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.SEARCH_NUM_TOKENS = [5, 10, 25, 50, 100, 150, 200]
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.DEEP = True # "whether do deep QKV or not, only for prepend location"
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.DROPOUT = 0.1
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.LAYER_BEHIND = True # True: prompt + layer; False: layer + prompt
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.SHARE_PARAM_KV = True # change it to False to init two parameters
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.ORIGIN_INIT = 2 # 0 for default, 1 for trunc_norm, 2 for kaiming init 
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.SHARED_ACCROSS = False # share vk value accross multi-attn layers

# Turn it to False when considering without MASK_CLS_TOKEN
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.MASK_CLS_TOKEN = True # set as the MAIN trigger to all cls token masked program(prouning and rewind process).
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.NORMALIZE_SCORES_BY_TOKEN = False # new added for normalized token (apply as xprompt)
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK = True # new added for cls token mask (own or disown whole prompt)
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK_PERCENT_NUM = None # set specific num of percent to mask
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK_PERCENT = [10, 20, 30, 40, 50, 60, 70, 80, 90] # percentage applied during selected
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.MIN_NUMBER_CLS_TOKEN = 1 # set the lower boundary to avoid overkilled

_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK_PIECES = True # new added for individual cls token mask (made pieces)
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_PIECE_MASK_PERCENT_NUM = None # set specific num of percent to mask
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_PIECE_MASK_PERCENT = [10, 20, 30, 40, 50, 60, 70, 80, 90] # percentage applied during selected
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.MIN_NUMBER_CLS_TOKEN_PIECE = 4 # set the lower boundary to avoid overkilled

_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_P_PIECES_NUM = 16 # new added to devided the pieces of token(for cls_token temporarily) 16
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.MASK_RESERVE = False # reserve the order of mask or not.

_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.REWIND_MASK_CLS_TOKEN_NUM = -1 # change correpsondingly during train
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.REWIND_MASK_CLS_TOKEN_PIECE_NUM = -1 # change correpsondingly during train
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.REWIND_STATUS = False # status mark for rewind process
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.REWIND_OUTPUT_DIR = ""
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.SAVE_REWIND_MODEL = False 

# Based on MASK_CLS_TOKEN == True
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.MASK_CLS_TOKEN_ON_VK = False # mask value and key instead of cls_token (unfinished, does not make sense)
# _GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.MASK_QUERY = False # does not make sense (attention map size changed, should apply linear projection to reduce dim)
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.QUERY_PROMPT_MODE = 0 # 0, 1, 2, 3 (disable query-prompt/ query transpose/query-key prompt(dim=3)/query-key-value prompt (2-dimension prompt))
_GLOBAL_CONFIG_FILE.MODEL.E2VPT.KV_PROMPT.FT_PT_MIXED = False # if true, will do mixed pt+ft training
#
#
# ----------------------------------------------------------------------
# 
# ----------------------------------------------------------------------
#
#
def get_cfg():
    """
    Get a copy of the default config.
    """
    return _GLOBAL_CONFIG_FILE.clone()