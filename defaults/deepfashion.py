from yacs.config import CfgNode as CN

_C = CN()

_C.ACCELERATE = CN()
_C.ACCELERATE.PROJECT_NAME = "CFLD"
_C.ACCELERATE.RUN_NAME = "inference"
_C.ACCELERATE.MIXED_PRECISION = "fp16"
_C.ACCELERATE.ALLOW_TF32 = True
_C.ACCELERATE.SEED = 3407
_C.ACCELERATE.GRADIENT_ACCUMULATION_STEPS = 1
_C.ACCELERATE.LOG_PERIOD = 10
_C.ACCELERATE.EVAL_PERIOD = 5

_C.MODEL = CN()
_C.MODEL.PRETRAINED_PATH = ""
_C.MODEL.LAST_EPOCH = 0
_C.MODEL.U_COND_PERCENT = 0.2
_C.MODEL.U_COND_DOWN_BLOCK_GUIDANCE = False
_C.MODEL.U_COND_UP_BLOCK_GUIDANCE = False

_C.MODEL.FIRST_STAGE_CONFIG = CN()
_C.MODEL.FIRST_STAGE_CONFIG.PRETRAINED_PATH = "pretrained_models/vae"

_C.MODEL.UNET_CONFIG = CN()
_C.MODEL.UNET_CONFIG.PRETRAINED_PATH = "pretrained_models/unet"
_C.MODEL.UNET_CONFIG.TRAINABLE_BLOCK_IDX = [11, 10, 9, 8, 7, 6, 5, 4, 3]
_C.MODEL.UNET_CONFIG.TRAIN_SELF_ATTN_Q = False
_C.MODEL.UNET_CONFIG.TRAIN_SELF_ATTN_K = False
_C.MODEL.UNET_CONFIG.TRAIN_SELF_ATTN_V = False
_C.MODEL.UNET_CONFIG.TRAIN_CROSS_ATTN_Q = False
_C.MODEL.UNET_CONFIG.TRAIN_CROSS_ATTN_K = True
_C.MODEL.UNET_CONFIG.TRAIN_CROSS_ATTN_V = True

_C.MODEL.SCHEDULER_CONFIG = CN()
_C.MODEL.SCHEDULER_CONFIG.NAME = "ddpm"
_C.MODEL.SCHEDULER_CONFIG.PRETRAINED_PATH = "pretrained_models/scheduler"
_C.MODEL.SCHEDULER_CONFIG.CUBIC_SAMPLING = True

_C.MODEL.COND_STAGE_CONFIG = CN()
_C.MODEL.COND_STAGE_CONFIG.PRETRAINED_PATH = "pretrained_models/swin/swin_base_patch4_window12_384_22kto1k.pth"
_C.MODEL.COND_STAGE_CONFIG.EMBED_DIM = 128
_C.MODEL.COND_STAGE_CONFIG.DEPTHS = [2, 2, 18, 2]
_C.MODEL.COND_STAGE_CONFIG.NUM_HEADS = [4, 8, 16, 32]
_C.MODEL.COND_STAGE_CONFIG.WINDOW_SIZE = 16
_C.MODEL.COND_STAGE_CONFIG.DROP_PATH_RATE = 0.2
_C.MODEL.COND_STAGE_CONFIG.LAST_NORM = False

_C.MODEL.APPEARANCE_GUIDANCE_CONFIG = CN()
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.CONVIN_KERNEL_SIZE = [1, 1, 1, 1, 1, 1, 1, 1, 1]
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.CONVIN_STRIDE = [1, 1, 1, 1, 1, 1, 1, 1, 1]
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.CONVIN_PADDING = [0, 0, 0, 0, 0, 0, 0, 0, 0]
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.ATTN_RESIDUAL_BLOCK_IDX = [11, 10, 9, 8, 7, 6, 5, 4, 3]
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.INNER_DIMS = [128, 128, 128, 256, 256, 256, 512, 512, 512]
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.CTX_DIMS = [320, 320, 320, 640, 640, 640, 1280, 1280, 1280]
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.EMBED_DIMS = [64, 64, 64, 128, 128, 128, 256, 256, 256]
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.HEADS = [2, 2, 2, 4, 4, 4, 8, 8, 8]
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.DEPTH = 4
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.TO_SELF_ATTN = False
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.TO_QUERIES = True
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.TO_KEYS = False
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.TO_VALUES = False
_C.MODEL.APPEARANCE_GUIDANCE_CONFIG.DETACH_INPUT = False

_C.MODEL.POSE_GUIDANCE_CONFIG = CN()
_C.MODEL.POSE_GUIDANCE_CONFIG.DOWNSCALE_FACTOR = 4
_C.MODEL.POSE_GUIDANCE_CONFIG.POSE_CHANNELS = 21
_C.MODEL.POSE_GUIDANCE_CONFIG.IN_CHANNELS = 320
_C.MODEL.POSE_GUIDANCE_CONFIG.CHANNELS = [320, 640, 1280]

_C.MODEL.DECODER_CONFIG = CN()
_C.MODEL.DECODER_CONFIG.N_CTX = 16
_C.MODEL.DECODER_CONFIG.CTX_DIM = 768
_C.MODEL.DECODER_CONFIG.DEPTH = 8
_C.MODEL.DECODER_CONFIG.HEADS = 24
_C.MODEL.DECODER_CONFIG.POSE_QUERY = False

_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = "adam"
_C.OPTIMIZER.EPOCHS = 100
_C.OPTIMIZER.WARMUP_STEPS = 1000
_C.OPTIMIZER.DECAY_EPOCHS = [50]
_C.OPTIMIZER.LR = 1.0e-4
_C.OPTIMIZER.SCALE_LR = False
_C.OPTIMIZER.WARMUP_RATE = 0.1
_C.OPTIMIZER.DECAY_RATE = 0.1
_C.OPTIMIZER.OVERRIDE_LR = 0.

_C.INPUT = CN()
_C.INPUT.ROOT_DIR = "./fashion"
_C.INPUT.BATCH_SIZE = 224
_C.INPUT.NUM_WORKERS = 8

_C.INPUT.GT = CN()
_C.INPUT.GT.IMG_SIZE = [512, 512]

_C.INPUT.COND = CN()
_C.INPUT.COND.IMG_SIZE = [256, 256]
_C.INPUT.COND.PRED_ASPECT_RATIO = [0.3, 1/0.3]
_C.INPUT.COND.PRED_RATIO = []
_C.INPUT.COND.PRED_RATIO_VAR = []
_C.INPUT.COND.MASK_PATCH_SIZE = 8
_C.INPUT.COND.MIN_SCALE = 1.0

_C.INPUT.POSE = CN()
_C.INPUT.POSE.IMG_SIZE = [256, 256]

_C.TEST = CN()
_C.TEST.NUM_INFERENCE_STEPS = 50
_C.TEST.MICRO_BATCH_SIZE = 16
_C.TEST.NUM_WORKERS = 8
_C.TEST.IMG_SIZE = [256, 176]

_C.TEST.DDIM_INVERSION_STEPS = 0
_C.TEST.DDIM_INVERSION_DOWN_BLOCK_GUIDANCE = False
_C.TEST.DDIM_INVERSION_UP_BLOCK_GUIDANCE = False
_C.TEST.DDIM_INVERSION_UNCONDITIONAL = True

# "uc_full", "updown_full", "down_full", "uc_down_full", "uc_down_updown_cdown", "uc_down_updown_full"
_C.TEST.GUIDANCE_TYPE = "uc_down_full"
_C.TEST.GUIDANCE_SCALE = 2.0
_C.TEST.DOWN_BLOCK_GUIDANCE_SCALE = 2.0
_C.TEST.UP_BLOCK_GUIDANCE_SCALE = 2.0
_C.TEST.ALL_BLOCK_GUIDANCE_SCALE = 2.0
_C.TEST.FULL_GUIDANCE_SCALE = 2.0
