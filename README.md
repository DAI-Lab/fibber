<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
[![PyPI Shield](https://img.shields.io/pypi/v/fibber.svg)](https://pypi.python.org/pypi/fibber)
[![Downloads](https://pepy.tech/badge/fibber)](https://pepy.tech/project/fibber)
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/fibber.svg?branch=stable&status=started)](https://travis-ci.org/DAI-Lab/fibber)
[![Coverage Status](https://codecov.io/gh/DAI-Lab/fibber/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/fibber)


# Fibber

Fibber is a library to evaluate different strategies to paraphrase natural language, especially how these strategies can break text classifiers without changing the meaning of a sentence.

- Documentation: [https://DAI-Lab.github.io/fibber](https://DAI-Lab.github.io/fibber)
- GitHub: [https://github.com/DAI-Lab/fibber](https://github.com/DAI-Lab/fibber)

# Overview

Fibber is a library to evaluate different strategies to paraphrase natural language. In this library, we have several built-in paraphrasing strategies. We also have a benchmark framework to evaluate the quality of paraphrase. In particular, we use the GPT2 language model to measure how meaningful is the paraphrased text. We use a universal sentence encoder to evaluate the semantic similarity between original and paraphrased text. We also train a BERT classifier on the original dataset, and check of paraphrased sentences can break the text classifier.

# Install

## Requirements

**fibber** has been developed and tested on [Python 3.6, 3.7 and 3.8](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of [conda](https://docs.conda.io/en/latest/miniconda.html)
is highly recommended to avoid interfering with other software installed in the system
in which **fibber** is run.

These are the minimum commands needed to create a conda environment using python3.6 for **fibber**:

```bash
# First you should install conda.
conda create -n fibber_env python=3.6
```

Afterward, you have to execute this command to activate the environment:

```bash
conda activate fibber_env
```

**Then you should install tensorflow and pytorch.** Please follow the instructions for [tensorflow](https://www.tensorflow.org/install) and [pytorch](https://pytorch.org). Fibber requires `tensorflow>=2.0.0` and `pytorch>=1.5.0`.


Remember to execute `conda activate fibber_env` every time you start a new console to work on **fibber**!



## Install from PyPI

After creating the conda environment and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **fibber**:

```bash
pip install fibber
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).

## Use without install

If you are using this project for research purpose and want to make changes to the code,
you can install all requirements by

```bash
git clone git@github.com:DAI-Lab/fibber.git
cd fibber
pip install --requirement requirement.txt
```

Then you can use fibber by

```base
python -m fibber.datasets.download_datasets
python -m fibber.benchmark.benchmark
```

In this case, any changes you made on the code will take effect immediately.


## Install from source

With your conda environment activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/fibber.git
cd fibber
git checkout stable
make install
```


# Quickstart

In this short tutorial, we will guide you through a series of steps that will help you
getting started with **fibber**.

**(1) [Install Fibber](#Install)**

**(2) Get a demo dataset.**

```python
from fibber.datasets import get_demo_dataset

trainset, testset = get_demo_dataset()
```

**(3) Create a Fibber object.**

```python
from fibber.fibber import Fibber

arg_dict = {
    "use_gpu_id": 0,
    "gpt2_gpu_id": 0,
    "strategy_gpu_id": 0,
}

fibber = Fibber(arg_dict, dataset_name="demo", strategy_name="RandomStrategy",
                trainset=trainset, testset=testset)
```

**(4) Randomly sample a sentence from the test set, and paraphrase it.**

The following command can randomly paraphrase the sentence into 5 different ways.

```python
fibber.paraphrase_a_random_sentence(n=5)
```

The output is a tuple of (str, list, list).

```python
# Original Text
'the movie slides downhill as soon as macho action conventions assert themselves .'

# 5 paraphrases
 ['conventions slides as as action assert macho downhill soon movie . the themselves',
  'as . downhill action macho the themselves assert as slides conventions soon movie',
  'movie as slides macho action . soon themselves the downhill as assert conventions',
  'the soon assert as movie themselves macho conventions as downhill . action slides',
  'downhill movie conventions slides the assert themselves action macho as as . soon'],

# Evaluation metrics of these 5 paraphrases.
 [{'EditingDistance': 8,
   'USESemanticSimilarity': 0.8859144449234009,
   'GloVeSemanticSimilarity': 1.0000000321979126,
   'GPT2GrammarQuality': 23.059619903564453},
  {'EditingDistance': 9,
   'USESemanticSimilarity': 0.8609699010848999,
   'GloVeSemanticSimilarity': 1.0000000321979126,
   'GPT2GrammarQuality': 39.824188232421875},
  {'EditingDistance': 8,
   'USESemanticSimilarity': 0.8530778288841248,
   'GloVeSemanticSimilarity': 1.0000000321979126,
   'GPT2GrammarQuality': 17.592607498168945},
  {'EditingDistance': 9,
   'USESemanticSimilarity': 0.8957847356796265,
   'GloVeSemanticSimilarity': 1.0000000321979126,
   'GPT2GrammarQuality': 24.76700210571289},
  {'EditingDistance': 9,
   'USESemanticSimilarity': 0.9004875421524048,
   'GloVeSemanticSimilarity': 1.0000000321979126,
   'GPT2GrammarQuality': 11.36586856842041}]
```

**(5) You can also ask fibber to paraphrase your sentence.**


```python
fibber.paraphrase({"text0": "This movie is fantastic"}, "text0", 5)
```



# Supported strategies

In this version, we implement three strategies

- IdentityStrategy:
	- The identity strategy outputs the original text as its paraphrase.
	- This strategy generates exactly 1 paraphrase for each original text regardless of `--num_paraphrases_per_text` flag.
- RandomStrategy:
	- The random strategy outputs the random shuffle of words in the original text.



# What's next?

For more details about **fibber** and all its possibilities
and features, please check the [documentation site](
https://DAI-Lab.github.io/fibber/).
