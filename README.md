# HTL-ReID

Official implementation of the manuscript "**Hierarchical Token Learning and Adaptive Gated Fusion for Robust Multi-Modal Object Re-Identification**".

## Overview

HTL-ReID is a unified framework for nighttime multi-modal (RGB / NIR / TIR) object re-identification. The current implementation extends the original HTL design into a quality-aware token selection and fusion pipeline:

- **Hierarchical Token Selection (HS)** — aggregates attention cues from shallow, middle, and deep ViT layers as complementary spatial priors, while keeping all token features in a single deep semantic space.
- **Dynamic FACSS** — jointly scores intra-modal discriminability and quality-weighted cross-modal cosine consensus, predicts a per-sample/per-modality token budget, and keeps a soft residual path outside hard top-k selection.
- **Nighttime modality quality estimation** — predicts RGB / NIR / TIR reliability with a night-scene prior so weak RGB, overexposed NIR, or noisy TIR evidence can be down-weighted per sample.
- **Quality-aware graph fusion** — replaces fixed rotation fusion with a fully connected RGB-NIR-TIR graph whose edge gates are conditioned on modality quality.
- **Modality adapters and local part branch** — adds lightweight modality-specific adapters after the shared ViT backbone and a selected-token part-level branch for local identity evidence.
- **Cross-modal auxiliary constraints** — adds quality-weighted alignment, token-consistency, and gate-balance losses during training.

HSL is not part of the current model path.

## Requirements
```bash
pip install -r requirements.txt
```

> `pytorch_wavelets` is vendored under `./pytorch_wavelets` — do **not** `pip install` it separately.

## Datasets
Download **RGBNT201**, **RGBNT100**, and **MSVR310** from their official sources, then update `DATASETS.ROOT_DIR` in the corresponding YAML config under `./configs/` to point to your local dataset directory.

## Pretrained Backbone
Download a pretrained ViT checkpoint and set `MODEL.PRETRAIN_PATH_T` in the YAML config to its local path. The backbone variant is selected via `MODEL.TRANSFORMER_TYPE`; supported values are listed in `modeling/make_model.py`.

## Training
Train on a single dataset by selecting its YAML config:

```bash
# RGBNT201
python train_net.py --config_file configs/RGBNT201/default.yml

# RGBNT100
python train_net.py --config_file configs/RGBNT100/default.yml

# MSVR310
python train_net.py --config_file configs/MSVR310/default.yml
```

The default training recipe uses AdamW, a smaller learning rate for the shared ViT backbone, higher-resolution inputs, grayscale patch replacement, modality dropout, ID + triplet supervision, part-branch supervision, and cross-modal auxiliary losses.

You can override any config field from the command line, e.g.:
```bash
python train_net.py --config_file configs/RGBNT201/default.yml \
    DATASETS.ROOT_DIR /path/to/datasets \
    MODEL.PRETRAIN_PATH_T /path/to/pretrained_vit.pth
```

## Evaluation
```bash
python test_net.py --config_file configs/RGBNT201/default.yml \
    TEST.WEIGHT /path/to/checkpoint.pth
```

`TEST.RE_RANKING` is enabled by default in the dataset YAML files for final evaluation.

## Smoke Test
Run the CPU-only pipeline test after installation or code changes:

```bash
python test_pipeline.py
```

The smoke test checks config merging, 3-modal and 2-modal forward/backward passes, save/load round-trip, and ablation switches without real datasets or pretrained weights.

## License
This project is released under the terms of the [LICENSE](LICENSE) file in this repository.
