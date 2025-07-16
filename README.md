# CRSLab
[![Pypi Latest Version](https://img.shields.io/pypi/v/crslab)](https://pypi.org/project/crslab)
[![Release](https://img.shields.io/github/v/release/rucaibox/crslab.svg)](https://github.com/rucaibox/crslab/releases)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-CRSLab-%23B21B1B)](https://arxiv.org/abs/2101.00939)
[![Documentation Status](https://readthedocs.org/projects/crslab/badge/?version=latest)](https://crslab.readthedocs.io/en/latest/?badge=latest)
[Paper](https://arxiv.org/pdf/2101.00939.pdf) | [Docs](https://crslab.readthedocs.io/en/latest/?badge=latest)
\| [中文版](./README_CN.md)

**CRSLab** is an open-source toolkit for building Conversational Recommender System (CRS). It is developed based on
Python and PyTorch. CRSLab has the following highlights:

* **Comprehensive benchmark models and datasets**: We have integrated commonly-used 6 datasets and 18 models, including graph neural network and pre-training models such as R-GCN, BERT and GPT-2. We have preprocessed these datasets to support these models, and release for downloading.
* **Extensive and standard evaluation protocols**: We support a series of widely-adopted evaluation protocols for testing and comparing different CRS.
* **General and extensible structure**: We design a general and extensible structure to unify various conversational recommendation datasets and models, in which we integrate various built-in interfaces and functions for quickly development.
* **Easy to get started**: We provide simple yet flexible configuration for new researchers to quickly start in our library.
* **Human-machine interaction interfaces**: We provide flexible human-machine interaction interfaces for researchers to conduct qualitative analysis.

<p align="center">
  <img src="https://i.loli.net/2020/12/30/6TPVG4pBg2rcDf9.png" alt="RecBole v0.1 architecture" width="400">
  <br>
  <b>Figure 1</b>: The overall framework of CRSLab
</p>

* [Installation](#Installation)
* [Quick-Start](#Quick-Start)
* [Models](#Models)
* [Datasets](#Datasets)
* [Performance](#Performance)
* [Releases](#Releases)
* [Contributions](#Contributions)
* [Citing](#Citing)
* [Team](#Team)
* [License](#License)

## Installation

CRSLab works with the following operating systems：
* Linux
* Windows 10
* macOS X

CRSLab requires Python version 3.7 or later.

CRSLab requires torch version 1.8. If you want to use CRSLab with GPU, please ensure that CUDA or CUDAToolkit version is 10.2 or later. Please use the combinations shown in this [Link](https://pytorch-geometric.com/whl/) to ensure the normal operation of PyTorch Geometric.

### Install PyTorch

Use PyTorch [Locally Installation](https://pytorch.org/get-started/locally/) or [Previous Versions Installation](https://pytorch.org/get-started/previous-versions/) commands to install PyTorch. For example, on Linux and Windows 10:

```bash
# CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# CPU Only
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch
```

If you want to use CRSLab with GPU, make sure the following command prints `True` after installation:

```bash
$ python -c "import torch; print(torch.cuda.is_available())"
>>> True
```

### Install PyTorch Geometric

Ensure that at least PyTorch 1.8.0 is installed:

```bash
$ python -c "import torch; print(torch.__version__)"
>>> 1.8.0
```

Find the CUDA version PyTorch was installed with:

```bash
$ python -c "import torch; print(torch.version.cuda)"
>>> 11.1
```

For Linux:
Install the relevant packages:
```
conda install pyg -c pyg
```

For others：
Check PyG [installation documents](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install the relevant packages.

### Install CRSLab

You can install from pip:
```bash
pip install crslab
```

OR install from source:
```bash
git clone https://github.com/RUCAIBox/CRSLab && cd CRSLab
pip install -e .
```

## Quick-Start

With the source code, you can use the provided script for initial usage of our library with cpu by default:

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml
```

The system will complete the data preprocessing, and training, validation, testing of each model in turn. Finally it will get the evaluation results of specified models.

If you want to save pre-processed datasets and training results of models, you can use the following command:

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml --save_data --save_system
```

In summary, there are following arguments in `run_crslab.py`:

* `--config` or `-c`: relative path for configuration file(yaml).
* `--gpu` or `-g`: specify GPU id(s) to use, we now support multiple GPUs. Defaults to CPU(-1).
* `--save_data` or `-sd`: save pre-processed dataset.
* `--restore_data` or `-rd`: restore pre-processed dataset from file.
* `--save_system` or `-ss`: save trained system.
* `--restore_system` or `-rs`: restore trained system from file.
* `--debug` or `-d`: use validation dataset to debug your system.
* `--interact` or `-i`: interact with your system instead of training.
* `--tensorboard` or `-tb`: enable tensorboard to monitor train performance.

## Models

In CRSLab, we unify the task description of conversational recommendation into three sub-tasks, namely recommendation (recommend user-preferred items), conversation (generate proper responses) and policy (select proper interactive action). The recommendation and conversation sub-tasks are the core of a CRS and have been studied in most of works. The policy sub-task is needed by recent works, by which the CRS can interact with users through purposeful strategy.

As the first release version, we have implemented 18 models in the four categories of CRS model, Recommendation model, Conversation model and Policy model.

|       Category       |                                                                                                                              Model                                                                                                                              |      Graph Neural Network?      |       Pre-training Model?       |
| :------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------: | :-----------------------------: |
|       CRS Model      | [ReDial](https://arxiv.org/abs/1812.07617)<br/>[KBRD](https://arxiv.org/abs/1908.05391)<br/>[KGSF](https://arxiv.org/abs/2007.04032)<br/>[TG-ReDial](https://arxiv.org/abs/2010.04125)<br/>[INSPIRED](https://www.aclweb.org/anthology/2020.emnlp-main.654.pdf) |    ×<br/>√<br/>√<br/>×<br/>×    |    ×<br/>×<br/>×<br/>√<br/>√    |
| Recommendation model |       Popularity<br/>[GRU4Rec](https://arxiv.org/abs/1511.06939)<br/>[SASRec](https://arxiv.org/abs/1808.09781)<br/>[TextCNN](https://arxiv.org/abs/1408.5882)<br/>[R-GCN](https://arxiv.org/abs/1703.06103)<br/>[BERT](https://arxiv.org/abs/1810.04805)       | ×<br/>×<br/>×<br/>×<br/>√<br/>× | ×<br/>×<br/>×<br/>×<br/>×<br/>√ |
|  Conversation model  |                                               [HERD](https://arxiv.org/abs/1507.04808)<br/>[Transformer](https://arxiv.org/abs/1706.03762)<br/>[GPT-2](http://www.persagen.com/files/misc/radford2019language.pdf)                                              |          ×<br/>×<br/>×          |          ×<br/>×<br/>√          |
|     Policy model     |                            PMI<br/>[MGCG](https://arxiv.org/abs/2005.03954)<br/>[Conv-BERT](https://arxiv.org/abs/2010.04125)<br/>[Topic-BERT](https://arxiv.org/abs/2010.04125)<br/>[Profile-BERT](https://arxiv.org/abs/2010.04125)                           |    ×<br/>×<br/>×<br/>×<br/>×    |    ×<br/>×<br/>√<br/>√<br/>√    |

Among them, the four CRS models integrate the recommendation model and the conversation model to improve each other, while others only specify an individual task.

For Recommendation model and Conversation model, we have respectively implemented the following commonly-used automatic evaluation metrics:

|        Category        |                                     Metrics                                     |
| :--------------------: | :-----------------------------------------------------------------------------: |
| Recommendation Metrics |                Hit@{1, 10, 50}, MRR@{1, 10, 50}, NDCG@{1, 10, 50}               |
|  Conversation Metrics  | PPL, BLEU-{1, 2, 3, 4}, Embedding Average/Extreme/Greedy, Distinct-{1, 2, 3, 4} |
|     Policy Metrics     |                              Accuracy, Hit@{1,3,5}                              |

## Datasets

We have collected and preprocessed 6 commonly-used human-annotated datasets, and each dataset was matched with proper KGs as shown below:

|                            Dataset                           | Dialogs | Utterances |    Domains   | Task Definition |  Entity KG |   Word KG  |
| :----------------------------------------------------------: | :-----: | :--------: | :----------: | :-------------: | :--------: | :--------: |
|        [ReDial](https://redialdata.github.io/website/)       |  10,006 |   182,150  |     Movie    |        --       |   DBpedia  | ConceptNet |
|      [TG-ReDial](https://github.com/RUCAIBox/TG-ReDial)      |  10,000 |   129,392  |     Movie    |   Topic Guide   | CN-DBpedia |   HowNet   |
|         [GoRecDial](https://arxiv.org/abs/1909.03922)        |  9,125  |   170,904  |     Movie    |  Action Choice  |   DBpedia  | ConceptNet |
|         [DuRecDial](https://arxiv.org/abs/2005.03954)        |  10,200 |   156,000  | Movie, Music |    Goal Plan    | CN-DBpedia |   HowNet   |
|      [INSPIRED](https://github.com/sweetpeach/Inspired)      |  1,001  |   35,811   |     Movie    | Social Strategy |   DBpedia  | ConceptNet |
| [OpenDialKG](https://github.com/facebookresearch/opendialkg) |  13,802 |   91,209   |  Movie, Book |  Path Generate  |   DBpedia  | ConceptNet |
|  [DistillRecDial](https://github.com/yourusername/DistillRecDial)  |   TBD   |     TBD     |     Movie    |        --       |   DBpedia  | ConceptNet |

## Performance

We have trained and test the integrated models on the TG-Redial dataset, which is split into training, validation and test sets using a ratio of 8:1:1. For each conversation, we start from the first utterance, and generate reply utterances or recommendations in turn by our model. We perform the evaluation on the three sub-tasks.

### Recommendation Task

|   Model   |    Hit\@1   |   Hit\@10  |   Hit\@50  |    MRR\@1   |   MRR\@10  |   MRR\@50  |   NDCG\@1   |  NDCG\@10  |  NDCG\@50  |
| :-------: | :---------: | :--------: | :--------: | :---------: | :--------: | :--------: | :---------: | :--------: | :--------: |
|   SASRec  |   0.000446  |   0.00134  |   0.0160   |   0.000446  |  0.000576  |   0.00114  |   0.000445  |   0.00075  |   0.00380  |
|  TextCNN  |   0.00267   |   0.0103   |   0.0236   |   0.00267   |   0.00434  |   0.00493  |   0.00267   |   0.00570  |   0.00860  |
|    BERT   |   0.00722   |   0.00490  |   0.0281   |   0.00722   |   0.0106   |   0.0124   |   0.00490   |   0.0147   |   0.0239   |
|    KBRD   |   0.00401   |   0.0254   |   0.0588   |   0.00401   |   0.00891  |   0.0103   |   0.00401   |   0.0127   |   0.0198   |
|    KGSF   |   0.00535   | **0.0285** | **0.0771** |   0.00535   |   0.0114   | **0.0135** |   0.00535   | **0.0154** | **0.0259** |
| TG-ReDial | **0.00793** |   0.0251   |   0.0524   | **0.00793** | **0.0122** |   0.0134   | **0.00793** |   0.0152   |   0.0211   |

### Conversation Task

|    Model    |  BLEU\@1  |  BLEU\@2  |   BLEU\@3  |   BLEU\@4  |  Dist\@1 |  Dist\@2 |  Dist\@3 |  Dist\@4 |  Average  |  Extreme  |   Greedy  |    PPL   |
| :---------: | :-------: | :-------: | :--------: | :--------: | :------: | :------: | :------: | :------: | :-------: | :-------: | :-------: | :------: |
|     HERD    |   0.120   |   0.0141  |   0.00136  |  0.000350  |   0.181  |   0.369  |   0.847  |   1.30   |   0.697   |   0.382   |   0.639   |    472   |
| Transformer |   0.266   |   0.0440  |   0.0145   |   0.00651  |   0.324  |   0.837  |   2.02   |   3.06   |   0.879   |   0.438   |   0.680   |   30.9   |
|     GPT2    |   0.0858  |   0.0119  |   0.00377  |   0.0110   | **2.35** | **4.62** | **8.84** | **12.5** |   0.763   |   0.297   |   0.583   |   9.26   |
|     KBRD    |   0.267   |   0.0458  |   0.0134   |   0.00579  |   0.469  |   1.50   |   3.40   |   4.90   |   0.863   |   0.398   |   0.710   |   52.5   |
|     KGSF    | **0.383** | **0.115** | **0.0444** | **0.0200** |   0.340  |   0.910  |   3.50   |   6.20   | **0.888** | **0.477** | **0.767** |   50.1   |
|  TG-ReDial  |   0.125   |   0.0204  |   0.00354  |  0.000803  |   0.881  |   1.75   |   7.00   |   12.0   |   0.810   |   0.332   |   0.598   | **7.41** |

### Policy Task

|    Model   |   Hit\@1  |  Hit\@10  |  Hit\@50  |   MRR\@1  |  MRR\@10  |  MRR\@50  |  NDCG\@1  |  NDCG\@10 |  NDCG\@50 |
| :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|    MGCG    |   0.591   |   0.818   |   0.883   |   0.591   |   0.680   |   0.683   |   0.591   |   0.712   |   0.729   |
|  Conv-BERT |   0.597   |   0.814   |   0.881   |   0.597   |   0.684   |   0.687   |   0.597   |   0.716   |   0.731   |
| Topic-BERT |   0.598   |   0.828   |   0.885   |   0.598   |   0.690   |   0.693   |   0.598   |   0.724   |   0.737   |
|  TG-ReDial | **0.600** | **0.830** | **0.893** | **0.600** | **0.693** | **0.696** | **0.600** | **0.727** | **0.741** |

The above results were obtained from our CRSLab in preliminary experiments. However, these algorithms were implemented and tuned based on our understanding and experiences, which may not achieve their optimal performance. If you could yield a better result for some specific algorithm, please kindly let us know. We will update this table after the results are verified.

## Releases

| Releases |      Date     |   Features   |
| :------: | :-----------: | :----------: |
|  v0.1.1  |  1 / 4 / 2021 | Basic CRSLab |
|  v0.1.2  | 3 / 28 / 2021 |    CRSLab    |

## Contributions

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/CRSLab/issues).
We welcome all contributions from bug fixes to new features and extensions.
We expect all contributions discussed in the issue tracker and going through PRs.
We thank the nice contributions through PRs from [@shubaoyu](https://github.com/shubaoyu), [@ToheartZhang](https://github.com/ToheartZhang).

## Citing

If you find CRSLab useful for your research or development, please cite our [Paper](https://arxiv.org/pdf/2101.00939.pdf):

```
@article{crslab,
    title={CRSLab: An Open-Source Toolkit for Building Conversational Recommender System},
    author={Kun Zhou, Xiaolei Wang, Yuanhang Zhou, Chenzhan Shang, Yuan Cheng, Wayne Xin Zhao, Yaliang Li, Ji-Rong Wen},
    year={2021},
    journal={arXiv preprint arXiv:2101.00939}
}
```

## Team

**CRSLab** was developed and maintained by [AI Box](http://aibox.ruc.edu.cn/) group in RUC.

## License

**CRSLab** uses [MIT License](./LICENSE).

## DistillRecDial Dataset and Environment Setup

We are excited to introduce the DistillRecDial dataset as part of the CRSLab framework. This dataset provides conversational recommendation data for improving recommendation and conversation tasks.

### Environment Setup

To run experiments with the DistillRecDial dataset, follow these steps:

1. Create a Python environment with Python 3.10 or later:
   ```bash
   python -m venv .env 
   source .env/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the experiments:
   ```bash
   python experiments.py
   ```

When running the experiments, the DistillRecDial dataset will be automatically downloaded and processed, similar to other datasets in the original CRSLab implementation.

### Reproducibility with Singularity Container

To enhance reproducibility, we've included a Singularity container definition file. To build and use the container:

1. Install Singularity on your system following the [official documentation](https://sylabs.io/guides/3.0/user-guide/installation.html).

2. Build the Singularity container:
   ```bash
   sudo singularity build crslab_container.sif singularity_CRSLab.def
   ```

3. Run experiments using the container:
   ```bash
   singularity run --nv crslab_container.sif python experiments.py
   ```

The `--nv` flag ensures that NVIDIA GPUs are accessible within the container if available on your system.

### Configuration Files

The configuration files for DistillRecDial are available in the repository. You can use them with the following commands:

```bash
# For recommendation task
python run_crslab.py --config config/crs/redial/distillrecdial.yaml

# For conversation task
python run_crslab.py --config config/conversation/gpt2/distillrecdial.yaml
```

### Evaluation on Recommendation, Conversation, and Conversational Recommendation

**Table 1: Recommendation metrics for various models.**

| Model                | Hit@1  | Hit@10 | Hit@50 | MRR@1  | MRR@10 | MRR@50 | NDCG@1 | NDCG@10 | NDCG@50 |
| -------------------- | :----: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: | :-----: |
| **Dialogue + Recommendation** |        |        |        |        |        |        |         |         |         |
| ReDial               | 0      | 0.0075 | 0.0367 | 0      | 0.0013 | 0.0024 | 0      | 0.0027  | 0.0088  |
| INSPIRED             | 0.0014 | 0.0090 | 0.0452 | 0.0014 | 0.0030 | 0.0045 | 0.0014 | 0.0043  | 0.0120  |
| KBRD                 | 0.0009 | 0.0104 | 0.0382 | 0.0009 | 0.0026 | 0.0036 | 0.0009 | 0.0043  | 0.0100  |
| **Recommendation Only** |        |        |        |        |        |        |         |         |         |
| BERT                 | 0.0763 | 0.1728 | 0.2763 | 0.0763 | 0.1039 | 0.1086 | 0.0763 | 0.1202  | 0.1428  |
| GRU4Rec              | 0      | 0.0013 | 0.0060 | 0      | 0.0001 | 0.0003 | 0      | 0.0004  | 0.0014  |
| SASRec               | 0.0040 | 0.0195 | 0.0209 | 0.0676 | 0.0040 | 0.0077 | 0.0098 | 0.0104  | 0.0208  |

**Table 2: Conversational (dialogue) metrics for various models. "--" indicates metric not reported.**

| Model                | BLEU@1 | BLEU@2 | BLEU@3 | BLEU@4 | Dist@1 | Dist@2 | Dist@3 | Dist@4 | F1     | Perplexity |
| -------------------- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----- | :--------: |
| **Dialogue + Recommendation** |        |        |        |        |        |        |        |        |        |            |
| ReDial               | 0.2232 | 0.0576 | 0.0322 | 0.0218 | 0.0071 | 0.0596 | 0.2415 | 0.5373 | 0.2232 |    204.7   |
| INSPIRED             | 0.7370 | 0.5788 | 0.4669 | 0.3825 | 0.0049 | 0.0760 | 0.4485 | 1.498  | 0.7495 |     2.198  |
| KBRD                 | 0.2986 | 0.1131 | 0.0601 | 0.0386 | 0.0072 | 0.1014 | 0.5029 | 1.281  | 0.3277 |     --     |
| **Dialogue Only** |        |        |        |        |        |        |        |        |        |            |
| GPT2                 | 0.7297 | 0.5659 | 0.4519 | 0.3672 | 0.0060 | 0.0789 | 0.4675 | 1.604  | 0.7421 |     5.136  |

--- 
If you use this version of CRSLab or these results, please cite
```bibtex
@inproceedings{Martina2025DistillRecDial,
  author    = {Martina, Alessandro Francesco Maria and Petruzzelli, Alessandro and Musto, Cataldo and de Gemmis, Marco and Lops, Pasquale and Semeraro, Giovanni},
  title     = {{DistillRecDial}: A Knowledge-Distilled Dataset Capturing User Diversity in Conversational Recommendation},
  booktitle = {Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys '25)},
  year      = {2025},
  month     = {September},
  day       = {22--26},
  address   = {Prague, Czech Republic},
  publisher = {ACM},
  doi       = {10.1145/3705328.3748161},
  isbn      = {979-8-4007-1364-4/2025/09}
}
```
