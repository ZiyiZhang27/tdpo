# Temporal Diffusion Policy Optimization (TDPO)

This is an official PyTorch implementation of **Temporal Diffusion Policy Optimization (TDPO)** from our paper [*Confronting Reward Overoptimization for Diffusion Models: A Perspective of Inductive and Primacy Biases*](https://arxiv.org/abs/2402.08552), which is accepted by **ICML 2024**.

## Installation
Python 3.10 or a newer version is required. In order to install the requirements, create a conda environment and run the `setup.py` file in this repository, e.g. run the following commands:

```bash
conda create -p tdpo python=3.10.12 -y
conda activate tdpo

git clone git@github.com:ZiyiZhang27/tdpo.git
cd tdpo
pip install -e .
```

## Training

To train on **Aesthetic Score** and evaluate *cross-reward generalization* by out-of-domain reward functions, run this command:

```bash
accelerate launch scripts/train_tdpo.py --config config/config_tdpo.py:aesthetic
```
To train on **PickScore** and evaluate *cross-reward generalization* by out-of-domain reward functions, run this command:

```bash
accelerate launch scripts/train_tdpo.py --config config/config_tdpo.py:pickscore
```

To train on **HPSv2** and evaluate *cross-reward generalization* by out-of-domain reward functions, run this command:

```bash
accelerate launch scripts/train_tdpo.py --config config/config_tdpo.py:hpsv2
```

For detailed explanations of all hyperparameters, please refer to the configuration files `config/base_tdpo.py` and `config/config_tdpo.py`. These files are pre-configured for training with 8 x NVIDIA A100 GPUs (each with 40GB of memory).

**Note:** Some hyperparameters might appear in both configuration files. In such cases, only the values set in `config/config_tdpo.py` will be used during training as this file has higher priority.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{zhang2024confronting,
  title={Confronting Reward Overoptimization for Diffusion Models: A Perspective of Inductive and Primacy Biases},
  author={Ziyi Zhang and Sen Zhang and Yibing Zhan and Yong Luo and Yonggang Wen and Dacheng Tao},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```

## Acknowledgement

- This repository is built upon the [PyTorch codebase of DDPO](https://github.com/kvablack/ddpo-pytorch) developed by Kevin Black and his team. We are grateful for their contribution to the field.

- We also extend our thanks to Timo Klein for open-sourcing the [PyTorch reimplementation](https://github.com/timoklein/redo/) of [ReDo](https://arxiv.org/abs/2302.12902).

- We also acknowledge the contributions of [PickScore](https://github.com/yuvalkirstain/PickScore), [HPSv2](https://github.com/tgxs002/HPSv2), and [ImageReward](https://github.com/THUDM/ImageReward) projects to this work.
