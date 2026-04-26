# HTL-ReID

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19769558.svg)](https://doi.org/10.5281/zenodo.19769558)

Official implementation of "**Hierarchical Token Learning and Adaptive Gated Fusion for Robust Multi-Modal Object Re-Identification**", submitted to *The Visual Computer*, 2026.

> ⚠️ This code is directly related to the manuscript submitted to *The Visual Computer*.
> If you use this code, please cite our paper.

## Overview

HTL-ReID is a unified framework for multi-modal (RGB / NIR / TIR) object re-identification. It combines three coordinated components:

- **Hierarchical Token Selection (HS)** — aggregates attention cues from shallow, middle, and deep ViT layers as complementary spatial priors, while keeping all token features in a single deep semantic space.
- **Fusion-Aware Synergistic Selection (FACSS)** — jointly scores intra-modal discriminability and cross-modal cosine consensus, modulated by an environment-aware dynamic weight.
- **Adaptive Gated Fusion (AGF)** — channel-wise convex interpolation gating that calibrates fusion intensity according to modality reliability.

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

## Citation
If you find this work useful, please cite:

```bibtex
@article{luo2026htl,
  title   = {Hierarchical Token Learning and Adaptive Gated Fusion for Robust Multi-Modal Object Re-Identification},
  author  = {Luo, Hongbin and Ye, Yihan},
  journal = {The Visual Computer},
  year    = {2026}
}
```

Please also cite the archived release:

```bibtex
@software{htl_reid_zenodo,
  title     = {HTL-ReID: Hierarchical Token Learning and Adaptive Gated Fusion for Robust Multi-Modal Object Re-Identification},
  author    = {Luo, Hongbin and Ye, Yihan},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19769558},
  url       = {https://doi.org/10.5281/zenodo.19769558}
}
```

## License
This project is released under the terms of the [LICENSE](LICENSE) file in this repository.
