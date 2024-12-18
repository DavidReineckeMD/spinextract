# RapidLymphoma: Fast intraoperative detection of primary CNS lymphoma and differentiation from common CNS tumors using stimulated Raman histology and deep learning

Implementation code for RapidLymphoma. Accepted in Neuro-Oncology.

[**Preprint on MedRxiv**](https://www.medrxiv.org/content/10.1101/2024.08.25.24312509v1)

[**Final Article available in Neuro-Oncology**](https://academic.oup.com/neuro-oncology/advance-article-abstract/doi/10.1093/neuonc/noae270/7924181?redirectedFrom=fulltext&login=true)



## Clinical Impact
This study introduces RapidLymphoma, a novel self-supervised-based deep-learning model for visual representations to leverage the detection of primary CNS Lymphoma (PCNSL) and differentiation from common and rare CNS tumors using intraoperative label-free stimulated Raman histology. While PCNSL is rare, time-critical personalized treatment with fast intraoperative decision-making is needed. In an international multicentric clinical trial, RapidLymphoma first demonstrated its ability to detect and differentiate PCNSL from other CNS lesions with an overall diagnostic balanced accuracy of 97.81% ±0.91 compared to formalin-fixed paraffin-embedded diagnosis and is non-inferior to frozen section analysis. It provides near real-time intraoperative feedback and guidance to the surgeon, delivering a diagnosis in under three minutes. RapidLymphoma extracts key histomorphological features for detecting and differentiating CNS lesions by utilizing the benefits of intraoperative stimulated Raman histology. This guidance is critical in surgical decision-making and provide surgeons amd pathologists a visual prediction heatmap feedback, highlighting critical areas within each whole-slide image.

## Workflow

![Overview](/figures/workflow.png)


## Installation

1. Clone RapidLymphoma github repo
   ```console
   git clone git@github.com:MLNeurosurg/rapidlymphoma.git
   ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
3. Create conda environment
    ```console
    conda create -n rapidlymphoma python=3.9
    ```
4. Activate conda environment
    ```console
    conda activate rapidlymphoma
    ```
5. Install package and dependencies
    ```console
    <cd /path/to/rapidlymphoma/repo/dir>
    pip install -e .
    ```

## Directory organization
- rapidlymphoma: the library for training with RapidLymphoma
    - datasets: PyTorch datasets to work with the data release
    - losses: Normalized Similarity loss function for BYOL
    - models: PyTorch networks for training and evaluation
    - train: training and evaluation scripts
- README.md
- LICENSE

# Training / evaluation instructions

The code base is written using PyTorch Lightning, with custom network and
datasets.


## RapidLymphoma training with adapted BYOL algorithm
1. Download and uncompress the data.
2. Update the sample config file in `train/config/train_rapidlymphoma.yaml` with
    desired configurations.
3. Change directory to `train` and activate the conda virtual environment.
4. Use `train/train_contrastive.py` to start training:
    ```console
    python train_contrastive.py -c config/train_rapidlymphoma.yaml
    ```
5. To run linear or finetuning protocol, update the config file
    `train/config/train_finetune.yaml` and continue training using
    `train/train_finetune.py`:
    ```console
    python train_finetune.py -c config/train_finetune.yaml
    ```

## Cross entropy experiments (ablation study)
1. Download and uncompress the data.
2. Update the sample config file in `train/config/train_ce.yaml` with desired
    configurations.
3. Change directory to `train` and activate the conda virtual environment.
4. Use `train/train_ce.py` to start training:
    ```console
    python train_ce.py -c config/train_ce.yaml
    ```

## Model evaluation
1. Update the sample config file in `train/config/eval.yaml` with desired
    configurations, including the PyTorch Lightning checkpoint you would like
    to use.
2. Change directory to `train` and activate the conda virtual environment.
3. Use `train/train_ce.py` to start training:
    ```console
    python eval.py -c config/eval.yaml
    ```

## RapidLymphoma model weights
[**Download RapidLymphoma weights**](https://huggingface.co/DavidReineckeMD/rapidlymphoma/blob/main/rapidlymphoma_resnet50.pt)

## License Information
RapidLymphoma data is released under Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0), and the code is licensed under the MIT License.
See LICENSE for license information and THIRD\_PARTY for third party notices.
Python and PyTorch code structure is based on #MLNeurosurg/opensrh .
