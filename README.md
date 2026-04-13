<p align="center">
  <h2 align="center">[ACL 2026] VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning
</h2>
  <p align="center">
    </br>
        <a href="https://github.com/Mr-Loevan/VL-Calibration">
        <img src='https://img.shields.io/badge/Code-GitHub-green' alt='Code'></a>
        <a href="https://arxiv.org/abs/2604.09529">
        <img src='https://img.shields.io/badge/Paper-Arxiv-red' alt='Paper'></a>
        <a href="https://modelscope.cn/datasets/xiaowenyi/VL-Calibration-12K">
        <img src='https://img.shields.io/badge/Dataset-ModelScope-blue' alt='Dataset'></a>
        <a href="https://modelscope.cn/collections/xiaowenyi/VL-Calibration">
        <img src='https://img.shields.io/badge/Model-ModelScope-blue' alt='Model'></a>
  </p>
</p>

## Overview

VL-Calibration is a framework for improving LVLMs calibration and reasoning via decoupled verbalized confidence.

## Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Installation

```bash
# Clone the repository
git clone https://github.com/Mr-Loevan/VL-Calibration.git
cd VL-Calibration

# Create conda environment
conda create -n vl_calib python=3.11
conda activate vl_calib

# Install dependencies (Refer to EasyR1 installation)
pip install -r requirements.txt
pip install -e .
```


## Quick Start

```bash
# Run decouple calibration training
# Download VL-Calibration-12K
bash examples/decouple.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{xiao2026vlcalibration,
      title={VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning},
      author={Wenyi Xiao and Xinchi Xu and Leilei Gan},
      year={2026},
      eprint={2604.09529},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.09529},
}
```

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgments

- Built on top of [veRL](https://github.com/volcengine/verl) and [EasyR1](https://github.com/hiyouga/EasyR1), Efficient, Scalable, Multi-Modality RL Training Frameworks.
