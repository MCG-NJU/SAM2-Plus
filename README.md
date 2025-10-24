# 🎯 SAM 2++: Tracking Anything at Any Granularity

<p align="center">
  <a href="https://tracking-any-granularity.github.io/"><img src="https://img.shields.io/badge/🏠%20web-Homepage-blue.svg" alt="Homepage"></a>
  <a href="https://arxiv.org/abs/2510.18822"><img src="https://img.shields.io/badge/📜%20arXiv-2510.18822-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/MCG-NJU/SAM2-Plus"><img src="https://img.shields.io/badge/🤗%20Model-SAM2--Plus-4dc0b0" alt="Model"></a>
  <a href="https://huggingface.co/datasets/MCG-NJU/Tracking-Any-Granularity"><img src="https://img.shields.io/badge/🤗%20Dataset-Tracking--Any--Granularity-ffca28" alt="Dataset"></a>
  <a href="TODO"><img src="https://img.shields.io/badge/🏆%20Leaderboard-Ranking-8b5cf6" alt="Leaderboard"></a>
</p>

[Jiaming Zhang](https://scholar.google.com/citations?hl=en&user=0lLB3fsAAAAJ), Cheng Liang, Yichun Yang, Chenkai Zeng,<br> [Yutao Cui](https://scholar.google.com/citations?user=TSMchWcAAAAJ&hl=en&oi=ao), Xinwen Zhang, Xin Zhou, Kai Ma, Gangshan Wu, [Limin Wang](http://wanglimin.github.io/)

**[Multimedia Computing Group, Nanjing University](http://mcg.nju.edu.cn/)**

## 🌟 Overview

![SAM 2 architecture](assets/1-overview.png?raw=true)

Video tracking aims at finding the specific target in subsequent frames given its initial state. Due to the varying granularity of target states across different tasks, most existing trackers are tailored to a single task and heavily rely on custom-designed modules within the individual task, which limits their generalization and leads to redundancy in both model design and parameters. To unify video tracking tasks, we present **SAM 2++**, a unified model towards tracking at any granularity, including masks, boxes, and points. First, to extend target granularity, we design task-specific prompts to encode various task inputs into general prompt embeddings, and a unified decoder to unify diverse task results into a unified form pre-output. Next, to satisfy memory matching, the core operation of tracking, we introduce a task-adaptive memory mechanism that unifies memory across different granularities. Finally, we introduce a customized data engine to support tracking training at any granularity, producing a large and diverse video tracking dataset with rich annotations at three granularities, termed **Tracking-Any-Granularity**, which represents a comprehensive resource for training and benchmarking on unified tracking. Comprehensive experiments on multiple benchmarks confirm that SAM 2++ sets a new state of the art across diverse tracking tasks at different granularities, establishing a unified and robust tracking framework.

## 🏗️ SAM 2++ Model

![SAM2++ model](assets/2-model.png?raw=true)

- We present a unified video tracking framework, termed as **SAM 2 ++**, which extends the SAM 2 model to track any targets in videos at any granularity, including masks, bounding boxes, and points.
- Due to the various task granularities, we introduce **task-specific prompts** to unify task input in different granularities and the **Unified Decoder** to unify diverse task results into a unified form pre-output.
- During mixture training, we found that a fully parameter-shared model training results in performance degradation due to the diverse memory requirements across tasks. To address this, we introduce a **task-adaptive memory mechanism** that dynamically adjusts memory representations according to each task's demand, enhancing the multi-task processing capability.

## 🗃️ Tracking-Any-Granularity Dataset

- We developed a comprehensive dataset for training our unified model, termed **T**racking-**A**ny-**G**ranularity (TAG), with annotations across three granularities: *segmentation masks, bounding boxes, and key points*. You can find some sample video sequences from the TAG dataset below (better view more samples in [project page](https://tracking-any-granularity.github.io/)):

  <table  align="center">
    <tbody>
      <tr>
        <td><img  width="220" src="assets/data/00025.gif"/></td>
        <td><img  width="220" src="assets/data/00076.gif"/></td>
        <td><img  width="220" src="assets/data/00045.gif"/></td>
      </tr>
    </tbody>
  </table>

  <table  align="center">
    <tbody>
      <tr>
        <td><img  width="220" src="assets/data/00102.gif"/></td>
        <td><img  width="220" src="assets/data/00103.gif"/></td>
        <td><img  width="220" src="assets/data/00152.gif"/></td>
      </tr>
    </tbody>
  </table>

  <table  align="center">
    <tbody>
      <tr>
        <td><img  width="220" src="assets/data/00227.gif"/></td>
        <td><img  width="220" src="assets/data/00117.gif"/></td>
        <td><img  width="220" src="assets/data/00312.gif"/></td>
      </tr>
    </tbody>
  </table>

- Our dataset includes **a wide range of video sources**, demonstrating **strong diversity** and serving as a solid benchmark for evaluating tracking performance. Each video sequence is annotated with **18 attributes** representing different tracking challenges, which can appear simultaneously in the same video. Common challenges include motion blur, deformation, and partial occlusion, reflecting the dataset’s high difficulty. Most videos contain multiple attributes, indicating the dataset’s coverage of complex and diverse tracking scenarios.

![TAG dataset](assets/4-attr.png?raw=true)

- The dataset has been released on [Hugging Face](https://huggingface.co/datasets/MCG-NJU/Tracking-Any-Granularity) and can be downloaded using the following code:

```bash
pip install huggingface_hub[cli]
huggingface-cli download MCG-NJU/Tracking-Any-Granularity --repo-type dataset --local-dir ../Tracking-Any-Granularity --local-dir-use-symlinks False --max-workers 16
```

## 🔥 Latest News

- **[2024-10-24]** SAM 2++ model and part of Tracking-Any-Granularity dataset are released. Check out the [project page](https://tracking-any-granularity.github.io/) for more details.

## 📑 Todo List

- [ ] Challenge Leaderboard for Tracking-Any-Granularity dataset
- [ ] Upload model to Hugging Face Model Hub
- [ ] Interactive Demo

## 🛠️ Installation

- Clone the repo:

```bash
git clone https://github.com/MCG-NJU/SAM2-Plus.git
cd SAM2-Plus
```

- Install the required packages:

```bash
# The code requires `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`.

conda create -n sam2_plus python=3.10 -y
conda activate sam2_plus

export PYTHONPATH=$PYTHONPATH:$(pwd)

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # torch 2.5.1 with cuda 12.1 for example

pip install -r sam2_plus/requirements.txt
pip install -r sav_dataset/requirements.txt

pip install -e .

pip install -e ".[dev]"
#pip install -e ".[notebooks]"
#pip install -e ".[interactive-demo]"

python setup.py build_ext --inplace
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); from sam2 import _C"
```

- Our project is developed based on SAM 2, so if you have any questions about installation, you can go to its [F&Q section](https://github.com/facebookresearch/sam2#installation) for answers. 
- Of course, you can also ask questions in our issue section, and we'll be happy to answer them.

## ▶️ Getting Started

### 📥 Download Checkpoints

First, we need to download checkpoint from [huggingface](https://huggingface.co/MCG-NJU/SAM2-Plus) with the script below to download all checkpoints:

```bash
pip install huggingface_hub[cli]
huggingface-cli download MCG-NJU/SAM2-Plus --local-dir ./checkpoints/SAM2-Plus --local-dir-use-symlinks False --max-workers 16
```

## 🪄 Inference SAM 2++

We provide an example script for running inference with SAM 2++ on our Tracking-Any-Granularity dataset. You can run the following command to test the model on a video sequence with different granularities: 
- [Video Object Segmentation (Mask Granularity)](tools/vos_inference_plus.sh)
- [Video Object Tracking (Box Granularity)](tools/sot_inference_plus.sh)
- [Point Tracking (Point Granularity)](tools/pt_inference_plus.sh).

## 🚀 Training SAM 2++

You can train or fine-tune SAM 2++ on datasets containing different granularities.

- SAM 2 has provided train file with `submitit` for cluster training, you can train our model in the same way:
```bash
python training/train.py \
--git-version <git-version> \
--config-module sam2_plus \
-c <train-config-path> \
--use-cluster 0 \
--num-gpus <num-gpus>
```

- Alternatively, in order to implement multi-machine training on ordinary machines, we implemented a training framework based on `torchrun` with `DistributedDataParallel (DDP)`. You can use the following command to start training:
```bash
torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=${NODE_RANK} \
training/train_ddp.py \
    --git-version <git-version> \
    --config-module sam2_plus \
    -c <train-config-path> \
    --torchrun_with_ddp
```

## 💥 Results

You can find some visualization results of SAM 2++ on different tracking tasks below (better view results in [project page](https://tracking-any-granularity.github.io/)):

  <table  align="center">
    <tbody>
      <tr>
        <th class="media1">Tracking-Any-Granularity</th>
        <th class="media1">MOSE</th>
        <th class="media1">VISOR</th>
      </tr>
      <tr>
        <td><img  width="220" src="assets/demo/mask-TA_val-00211.gif"/></td>
        <td><img  width="220" src="assets/demo/mask-MOSE-ba5644c3--6.gif"/></td>
        <td><img  width="220" src="assets/demo/mask-VISOR_val-P24_09_seq_00055--3.gif"/></td>
      </tr>
    </tbody>
  </table>

  <table  align="center">
    <tbody>
      <tr>
        <th class="media1">Tracking-Any-Granularity</th>
        <th class="media1">GOT-10k</th>
        <th class="media1">NFS</th>
      </tr>
      <tr>
        <td><img  width="220" src="assets/demo/box-TA_val-00721.gif"/></td>
        <td><img  width="220" src="assets/demo/box-Got10K-GOT-10k_Test_000143--15.gif"/></td>
        <td><img  width="220" src="assets/demo/box-NFS-nfs_soccer_player_2.gif"/></td>
      </tr>
    </tbody>
  </table>

  <table  align="center">
    <tbody>
      <tr>
        <th class="media1">Tracking-Any-Granularity</th>
        <th class="media1">TAPVid DAVIS</th>
        <th class="media1">RoboTAP</th>
      </tr>
      <tr>
        <td><img  width="220" src="assets/demo/points-TrackingAnything_val-00300--3--3--20--15.gif"/></td>
        <td><img  width="220" src="assets/demo/points-Tapvid_davis-00002--4--3--20--15.gif"/></td>
        <td><img  width="220" src="assets/demo/points-Tapvid_robotap-00055--2--2--20--15-resize.gif"/></td>
      </tr>
    </tbody>
  </table>

## 📄 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 👍 Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## 📚 Citing SAM 2++

If you use SAM 2++ or the Tracking-Any-Granularity dataset in your research, please use the following BibTeX entry.

```bibtex
@article{zhang2025sam2trackinggranularity,
  title={SAM 2++: Tracking Anything at Any Granularity},
  author={Jiaming Zhang and Cheng Liang and Yichun Yang and Chenkai Zeng and Yutao Cui and Xinwen Zhang and Xin Zhou and Kai Ma and Gangshan Wu and Limin Wang},
  journal={arXiv preprint arXiv:2510.18822},
  url={https://arxiv.org/abs/2510.18822},
  year={2025}
}
```

## 🙏 Acknowledgments
We would like to thank [Segment Anything 2 (SAM 2)](https://github.com/facebookresearch/segment-anything) for their contributions to the field of computer vision and for providing the foundation upon which SAM 2++ is built.