# Who knows best? Intelligent Crowdworker Selection via Deep Learning

Authors: Marek Herde, Denis Huseljic, Bernhard Sick, Ulrich Bretschneider, 
and Sarah Oeste-Reiß

## Project Structure
- [`evaluation`](/evaluation): collection of Python and Bash scripts required to perform experimental evaluation
- [`lfma`](/lfma): Python package consisting of several sub-packages
    - [`classifiers`](/lfma/classifiers): implementation of multi-annotator supervised learning techniques according to
      scikit-learn interfaces
    - [`modules`](/lfma/modules): implementation of multi-annotator supervised learning techniques as 
      [`pytorch_lightning`](https://www.pytorchlightning.ai/) modules, [`pytorch`](https://pytorch.org/) data sets, 
      and special layers
    - [`utils`](/lfma/utils): helper functions
- [`notebooks`](/notebooks):
  - [`annotator_simulation.ipynb`](/notebooks/annotator_simulation.ipynb): simulation of annotator sets for the data sets 
    LETTER and CIFAR10
  - [`data_set_creation_download.ipynb`](/notebooks/data_set_creation_download.ipynb): download of LETTER and CIFAR10
  - [`evaluation.ipynb`](/notebooks/evaluation.ipynb): loading and presentation of experimental results
- [`requirements.txt`](requirements.txt): list of Python packages required to reproduce experiments 

## How to execute experiments?
In the following, we describe step-by-step how to execute all experiments presented in the accompanied article. 
As a prerequisite, we assume to have a Linux distribution as operating system and 
[`conda`](https://docs.conda.io/en/latest/) installed on your machine.

1. _Setup Python environment:_
```bash
projectpath$ conda create --name crowd python=3.9
projectpath$ conda activate crowd
```
First, we need to install `torch` with the build (1.13.1). For this purpose, we refer to 
[`pytorch`](https://pytorch.org/). An exemplary command for a Linux operating system would be:
```bash
projectpath$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Subsequently, we install the remaining requirements:
```bash
projectpath$ pip install -r requirements.txt
```
2. _Create and download data sets:_ Start jupyter-notebook and follow the instructions in the jupyter-notebook file
[`notebooks/data_set_creation_download.ipynb`](/notebooks/data_set_creation_download.ipynb).
```bash
projectpath$ conda activate crowd
projectpath$ jupyter-notebook
```
3. _Simulate annotators:_ Start jupyter-notebook and follow the instructions in the jupyter-notebook file 
[`notebooks/annotator_simulation.ipynb`](/notebooks/annotator_simulation.ipynb).
```bash
projectpath$ conda activate crowd
projectpath$ jupyter-notebook
```
4. _Execute experiment scripts:_ The files 
`evaluation/letter.sh` and `evaluation/cifar10.sh` corresponds to evaluating MaDL on CIFAR10 and LETTER. Such a 
file consists of multiple commands executing the file [`evaluation/run_experiment.py`](evaluation/run_experiment.py) with 
different configurations. For a better understanding of these possible configurations, we refer to the explanations in 
the file [`evaluation/run_experiment.py`](evaluation/run_experiment.py). Further you need to specify certain paths, e.g., for logging before 
execution. You can now execute such a `bash` script via:
```bash
projectpath$ conda activate crowd
projectpath$ ./evaluation/crowd_letter.sh
projectpath$ ./evaluation/crowd_cifar10.sh
```
Alternatively, you can use the `sbatch` command:
```bash
projectpath$ conda activate crowd
projectpath$ sbatch ./evaluation/letter.sh
projectpath$ sbatch ./evaluation/crowd_cifar10.sh
```

## How to investigate the experimental results?
Once, an experiment is completed, its associated results are saved as a `.csv` file at the directory specified by 
`evaluation.run_experiment.RESULT_PATH`. For getting a summarized presentation of these results, you need 
to start jupyter-notebook and follow the instructions in the jupyter-notebook file 
[`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb).
```bash
projectpath$ conda activate crowd
projectpath$ jupyter-notebook
```

## References
The code is majorly based on and adopted from [Multi-annotator Deep Learning (MaDL)](https://github.com/ies-research/multi-annotator-deep-learning). 

## Citing
If you use this software in one of your research projects or would like to reference the 
accompanied article, please use the following:

```
@inproceedings{
    herde2023who,
    title={Who knows best? Intelligent Crowdworker Selection via Deep Learning},
    author={Marek Herde and Denis Huseljic and Bernhard Sick and Ulrich Bretschneider and Sarah Oeste-Rei{\ss}},
    booktitle={Interactive Adaptive Learning Workshop @ ECML/PKDD},
    pages={14--18},
    year={2023},
    url={https://ceur-ws.org/Vol-3470/paper3.pdf},
}
```
