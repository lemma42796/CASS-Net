from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of model
_C.MODEL.NAME = 'CASS-Net'
_C.MODEL.METHOD = 'CASS'
# Margin of triplet loss
_C.MODEL.MARGIN = 0
# Path to pretrained model of backbone. Supports local .pth/.bin/.safetensors
# files, checkpoint directories, and timm://model_name sources, e.g.
# timm://vit_base_patch16_224.augreg2_in21k_ft_in1k
_C.MODEL.PRETRAIN_PATH_T = 'your_path_to_pretrained_model'
# Optional full-model checkpoint used for staged CASS fine-tuning.
# Load happens before optimizer construction, so optimizer/scheduler start fresh.
_C.MODEL.RESUME_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# Choose different resnet backbone (1->resnet50||2->resnet101||3->resnet152)
# Fusion dim
_C.MODEL.MIX_DIM = 768
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# Choose the supervision type of the backbone
_C.MODEL.AL = 0
# [Deprecated] kept for backward-compatible cfg loading; replaced by HS_K
_C.MODEL.HEAD_KEEP = 1
# The keep tokens in the Frequency Selection Part
_C.MODEL.FREQUENCY_KEEP=10
_C.MODEL.FREQUENCY_QUALITY_AWARE = 1
# HS (Hierarchical Token Selection)
_C.MODEL.HS_LAYERS = [4, 8, 12]
_C.MODEL.HS_K = 16
# FACSS (Fusion-Aware Synergistic Selection)
_C.MODEL.FACSS_K = 16
_C.MODEL.FACSS_DYNAMIC_K = 1
_C.MODEL.FACSS_MIN_K = 8
_C.MODEL.FACSS_MAX_K = 32
_C.MODEL.FACSS_K_HIDDEN = 192
_C.MODEL.FACSS_SOFT_RESIDUAL_WEIGHT = 0.15
_C.MODEL.FACSS_ALPHA_HIDDEN = 192
_C.MODEL.SELECTED_PATCH_BLEND_WEIGHT = 0.15
_C.MODEL.SELECTED_PATCH_CONTEXT = 'mean'
_C.MODEL.SELECTED_PATCH_ATTN_SCALE = 10.0
_C.MODEL.SELECTED_PATCH_GATE_INIT = 0.0
# Pooling for cross-modal cosine: 'max' (paper) | 'topk' | 'lse'
_C.MODEL.FACSS_CROSS_POOL = 'max'
_C.MODEL.FACSS_CROSS_TOPK = 3
_C.MODEL.FACSS_CROSS_LSE_TAU = 5.0
# Alpha granularity: 'sample' (paper) | 'token'
_C.MODEL.FACSS_ALPHA_GRANULARITY = 'sample'
# Score normalization: 'minmax' (paper) | 'robust' | 'zscore'
_C.MODEL.FACSS_NORM = 'minmax'
# Straight-through estimator on top-K to enable gradient flow into alpha_mlp;
# forward is identical to hard top-K, backward routes through softmax of scores.
_C.MODEL.FACSS_STE = 1
_C.MODEL.FACSS_STE_TAU = 1.0
_C.MODEL.FACSS_MODALITY_UNION = 1
_C.MODEL.FACSS_UNION_PROMOTE = 1
# OCFR auxiliary loss (not in paper); off by default for paper-faithful reproduction
_C.MODEL.OCFR = 0
# Ablation switches (1=enable, 0=disable)
_C.MODEL.AGF = 0
# AGF (Adaptive Gated Fusion) hyperparameters
_C.MODEL.AGF_NUM_HEADS = 12
# Nighttime modality reliability. The prior only initializes the quality head:
# RGB is kept useful but starts slightly below NIR/TIR for night imagery.
_C.MODEL.QUALITY_AWARE = 0
_C.MODEL.QUALITY_HIDDEN = 192
_C.MODEL.QUALITY_PRIOR = [0.50, 0.65, 0.65]
_C.MODEL.QUALITY_MIN_SCORE = 0.05
# Lightweight modality adapters after the shared ViT backbone.
_C.MODEL.MODALITY_ADAPTER = 0
_C.MODEL.MODALITY_ADAPTER_DIM = 192
_C.MODEL.MODALITY_ADAPTER_SCALE = 0.5
# Local identity evidence from selected tokens.
_C.MODEL.PART_BRANCH = 0
_C.MODEL.PART_NUM = 3
# Auxiliary cross-modal constraints.
_C.MODEL.ALIGN_LOSS_WEIGHT = 0.0
_C.MODEL.TOKEN_CONSISTENCY_WEIGHT = 0.0
_C.MODEL.GATE_BALANCE_WEIGHT = 0.0
_C.MODEL.FUSE_LOSS_WEIGHT = 1.0
_C.MODEL.BRANCH_LOSS_WEIGHT = 0.5
_C.MODEL.PART_LOSS_WEIGHT = 0.25
_C.MODEL.AUX_LOSS_WEIGHT = 0.3
_C.MODEL.AUX_WARMUP_EPOCHS = 40

# CASS-Net (chapter 4): HSS + SQT + NGA + CA-GF
# Incremental ablation stage:
# 'baseline' | 'hss' | 'hss_nga' | 'hss_nga_cagf' |
# 'hss_sqt' | 'hss_sqt_nga' | 'hss_sqt_nga_cagf' | 'full'
_C.MODEL.CASS_ABLATION_STAGE = 'full'
_C.MODEL.CASS_DESCRIPTOR_MODE = 'summary'
_C.MODEL.CASS_CLS_CONTEXT_WEIGHT = 0.0
_C.MODEL.CASS_NUM_HEADS = 12
_C.MODEL.CASS_TOPK = 128
_C.MODEL.CASS_DYNAMIC_TOPK = 1
_C.MODEL.CASS_MIN_TOPK = 0
_C.MODEL.CASS_MAX_TOPK = 0
_C.MODEL.CASS_STE = 1
_C.MODEL.CASS_STE_TAU = 1.0
_C.MODEL.CASS_MODAL_ALPHA = 1
_C.MODEL.CASS_SELECTOR_SOFT_GATE = 1
_C.MODEL.CASS_SOFT_GATE_TAU = 0.25
_C.MODEL.CASS_SQT_FALLBACK_ALPHA = 0.4
_C.MODEL.CASS_SQT_NUM_QUERIES = 4
_C.MODEL.CASS_SQT_CLS_QUERY_WEIGHT = 0.5
_C.MODEL.CASS_SQT_CLS_SCORE_WEIGHT = 0.375
_C.MODEL.CASS_SQT_DIVERSITY_WEIGHT = 0.01
_C.MODEL.CASS_SQT_USE_SELECTOR = 1
_C.MODEL.CASS_SQT_FUSION_WEIGHT = 0.0
_C.MODEL.CASS_SQT_SUMMARY_TAU = 0.25
_C.MODEL.CASS_SQT_AGREEMENT_GATE = 0
_C.MODEL.CASS_SQT_MAX_RESIDUAL_NORM = 0.0
_C.MODEL.CASS_SQT_LEARNABLE_GATE = 0
_C.MODEL.CASS_SQT_GATE_INIT = -2.0
_C.MODEL.CASS_SQT_WARMUP_EPOCHS = 0
_C.MODEL.CASS_SQT_RAMP_EPOCHS = 0
_C.MODEL.CASS_SOFT_RESIDUAL_WEIGHT = 0.2
_C.MODEL.CASS_HSS_EDGES = 256
_C.MODEL.CASS_HSS_FILTERS = 128
_C.MODEL.CASS_HSS_GRAPH_WEIGHT = 1.0
_C.MODEL.CASS_HSS_GRAPH_WARMUP_EPOCHS = 0
_C.MODEL.CASS_HSS_THETA = 0.0
_C.MODEL.CASS_HSS_WHITEN = 1
_C.MODEL.CASS_HSS_RESIDUAL_ADAPTER = 1
_C.MODEL.CASS_HSS_ADAPTER_DIM = 192
_C.MODEL.CASS_HSS_ADAPTER_NORM = 1
_C.MODEL.CASS_HSS_GATE_INIT = 0.0
_C.MODEL.CASS_HSS_GATE_FLOOR = 0.0
_C.MODEL.CASS_HSS_GATE_FLOOR_WARMUP_EPOCHS = 3
_C.MODEL.CASS_HSS_SCORE_MIX = 0.0
_C.MODEL.CASS_HSS_SCORE_SOURCE = 'residual'
_C.MODEL.CASS_HSS_SCORE_DETACH = 0
_C.MODEL.CASS_WHITEN_GROUP_SIZE = 16
_C.MODEL.CASS_WHITEN_MOMENTUM = 0.1
_C.MODEL.CASS_WHITEN_EPS = 1e-3
_C.MODEL.CASS_NGA_MEMORY = 1
_C.MODEL.CASS_NGA_REFRESH_PERIOD = 1
_C.MODEL.CASS_NGA_KNN = 20
_C.MODEL.CASS_NGA_HIDDEN = 32
_C.MODEL.CASS_NGA_GATE_GROUPS = 16
_C.MODEL.CASS_NGA_GATE_SCALE = 0.1
_C.MODEL.CASS_NGA_RESIDUAL_WEIGHT = 0.0
_C.MODEL.CASS_NGA_RESIDUAL_MODE = 'sqt_gate'
_C.MODEL.CASS_NGA_QUERY_ANCHOR = 1
_C.MODEL.CASS_NGA_WARMUP_EPOCHS = 5
_C.MODEL.CASS_NGA_EMA_MOMENTUM = 0.5
_C.MODEL.CASS_NGA_USE_PROTOTYPE = 1
_C.MODEL.CASS_CAGF_MODE = 'agreement'
_C.MODEL.CASS_CAGF_RESIDUAL_WEIGHT = 0.10
_C.MODEL.CASS_CAGF_SELF_WEIGHT = 0.25
_C.MODEL.CASS_CAGF_MIN_AGREE = 0.30
_C.MODEL.CASS_CAGF_AGREE_TAU = 0.15
_C.MODEL.CASS_CAGF_MAX_GATE = 0.25
_C.MODEL.CASS_CAGF_MAX_RESIDUAL_NORM = 0.25
_C.MODEL.CASS_CAGF_DETACH_CONTEXT = 1
_C.MODEL.CASS_CAGF_WARMUP_EPOCHS = 5
_C.MODEL.CASS_CAGF_RAMP_EPOCHS = 3
_C.MODEL.CASS_QUALITY_AWARE = 1
_C.MODEL.CASS_PART_BRANCH = 1
_C.MODEL.CASS_PART_NUM = 3
_C.MODEL.CASS_FUSED_LOSS_WEIGHT = 1.0
_C.MODEL.CASS_MODAL_LOSS_WEIGHT = 0.25
_C.MODEL.CASS_PART_LOSS_WEIGHT = 0.5

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224'
# The stride size of the backbone
_C.MODEL.STRIDE_SIZE = [16, 16]

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = True
_C.MODEL.SIE_VIEW = False
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
# Value of padding size
_C.INPUT.PADDING = 10
_C.INPUT.GRAY_REPLACE_PROB = 0.3
_C.INPUT.MODALITY_DROP_PROB = 0.1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('RGBNT201')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')
# Number of modalities per training sample. RGBNT/MSVR use 3; SYSU-MM01 and
# RegDB use paired visible/infrared samples.
_C.DATASETS.MODALITIES = 3
# If enabled, validation/inference extracts single visible or infrared branch
# features using the modality marker stored in each sample.
_C.DATASETS.CROSS_MODAL_EVAL = False
_C.DATASETS.SYSU_EVAL_MODE = 'all'
_C.DATASETS.REGDB_SPLIT = 1
_C.DATASETS.REGDB_QUERY_MODALITY = 'thermal'
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 14
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "AdamW"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Optional training stop epoch. Keep 0 to train through MAX_EPOCHS while the
# scheduler still uses MAX_EPOCHS for its schedule.
_C.SOLVER.TRAIN_EPOCHS = 0
# Base learning rate
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.BACKBONE_LR_FACTOR = 0.1
_C.SOLVER.NEW_MODULE_LR_FACTOR = 1.0
# Factor of learning bias
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.05
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 20
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
# Scheduler step unit. CASS keeps the historical epoch schedule by default;
# HTL reproduction overlays can select iteration to match the HTL codebase.
_C.SOLVER.SCHEDULER_UNIT = "epoch"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.SEED = 1111
_C.SOLVER.AMP = True
# Mixed-precision dtype when SOLVER.AMP is enabled: 'bf16', 'fp16', or 'fp32'.
_C.SOLVER.AMP_DTYPE = 'bf16'
# Full training checkpoint for exact continuation. Unlike MODEL.RESUME_PATH,
# this restores model, optimizer, scheduler, scaler, epoch, RNG, and best metrics.
_C.SOLVER.RESUME_CHECKPOINT = ''
# Keep disk usage low by default. Enable this explicitly only when exact
# interruption recovery is more important than storage.
_C.SOLVER.SAVE_LATEST_CHECKPOINT = False
_C.MODEL.NO_MARGIN = True
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 60
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 10
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.KL = 0
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 128 
# A batch size of 128 yields better results for both person and vehicle datasets compared to 64. 
# Using a batch size of 64 may result in a slight decrease in performance.

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 64
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'yes'
_C.TEST.RERANK_K1 = 50
_C.TEST.RERANK_K2 = 15
_C.TEST.RERANK_LAMBDA = 0.3
# Refresh CASS NGA train memory before standalone inference. 'auto' refreshes
# only for CASS runs with MODEL.CASS_NGA_MEMORY enabled; use 'no' for legacy
# debugging.
_C.TEST.REFRESH_CASS_NGA_MEMORY = 'auto'
# -1 means refresh memory in the fully enabled inference state. Set a positive
# epoch only when reproducing a specific memory warmup/checkpoint context.
_C.TEST.CASS_NGA_MEMORY_EPOCH = -1
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'before'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# ----------------------------------------------------------a------------------ #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./outputs/CASS-Net"
