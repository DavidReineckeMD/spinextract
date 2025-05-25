# SpineXtract: Rapid intraoperative AI-driven diagnostics for spinal tumors using label-free optical imaging

Implementation code for SpineXtract.


[**Final Article available in XX**]



## Clinical Impact
SpineXtract addresses a critical gap in spinal tumor surgery through its ability to deliver rapid and accurate diagnostic insights that enhance intraoperative and clinical decision-making processes. By significantly reducing diagnostic timelines, this technology has the potential to shorten surgical duration and improve workflow efficiency. The system's robust performance across our international virtual trial demonstrates its adaptability to diverse clinical environments and delivers intraoperative visual feedback for immediate surgical guidance and further decision-making.

## Technical Impact
Our transformer-based classifier leverages advanced self-attention mechanisms within each image patch to optimize feature extraction and classification accuracy. The model partitions input embeddings into divisible segments, applying targeted attention to the most diagnostically relevant features within each patch. This domain-specific architectural adaptation enhances SpineXtract's ability to identify subtle histological patterns characteristic of different spinal tumor types, resulting in superior performance on this critical downstream clinical task compared to current state-of-the-art approaches.

## Workflow

![Overview](/figures/workflow.png)


## Installation

1. Clone spinextract github repo
   ```console
   git clone git@github.com:DavidReineckeMD/spinextract.git
   ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
3. Create conda environment
    ```console
    conda create -n spinextract python=3.9
    ```
4. Activate conda environment
    ```console
    conda activate spinextract
    ```
5. Install package and dependencies
    ```console
    <cd /path/to/spinextract/repo/dir>
    pip install -e .
    ```

## Directory organization
- spinextract: the library for training with spinextract
    - datasets: PyTorch datasets to work with the data release
    - losses: Normalized Similarity loss function for BYOL
    - models: PyTorch networks for training and evaluation
    - train: training and evaluation scripts
- README.md
- LICENSE

# Training / evaluation instructions

The code base is written using PyTorch Lightning, with custom network and
datasets.


## spinextract training with adapted BYOL algorithm
1. Download and uncompress the data.
2. Update the sample config file in `train/config/train_spinextract.yaml` with
    desired configurations.
3. Change directory to `train` and activate the conda virtual environment.
4. Use `train/train_contrastive.py` to start training:
    ```console
    python train_contrastive.py -c config/train_spinextract.yaml
    ```
5. To run linear, transformer or finetuning protocol, update the config file
    `train/config/train_finetune.yaml` and continue training using
    `train/train_finetune.py`:
    ```console
    python train_finetune.py -c config/train_finetune.yaml
    ```


## Model evaluation
1. Update the sample config file in `train/config/eval.yaml` with desired
    configurations, including the PyTorch Lightning checkpoint you would like
    to use.
2. Change directory to `eval` and activate the conda virtual environment.
3. Use `eval/eval.py` to start training:
    ```console
    python eval.py -c config/eval.yaml
    ```

## SpineXtract model weights
[**Download spinextract weights**](https://huggingface.co/DavidReineckeMD/spinextract)

## License Information
SpineXtract data is released under Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0), and the code is licensed under the MIT License.
See LICENSE for license information and THIRD\_PARTY for third party notices.
Python and PyTorch code structure is based on #MLNeurosurg/opensrh .
