<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
[![PyPI Shield](https://img.shields.io/pypi/v/fibber.svg)](https://pypi.python.org/pypi/fibber)
[![Downloads](https://pepy.tech/badge/fibber)](https://pepy.tech/project/fibber)
![build](https://github.com/DAI-Lab/fibber/workflows/build/badge.svg?branch=stable)

# Fibber

Fibber is a library to evaluate different strategies to paraphrase natural language, especially how these strategies can break text classifiers without changing the meaning of a sentence.

- Documentation: [https://DAI-Lab.github.io/fibber](https://DAI-Lab.github.io/fibber)
- GitHub: [https://github.com/DAI-Lab/fibber](https://github.com/DAI-Lab/fibber)

# Overview

Fibber is a library to evaluate different strategies to paraphrase natural language. In this library, we have several built-in paraphrasing strategies. We also have a benchmark framework to evaluate the quality of paraphrase. In particular, we use the GPT2 language model to measure how meaningful is the paraphrased text. We use a universal sentence encoder to evaluate the semantic similarity between original and paraphrased text. We also train a BERT classifier on the original dataset, and check of paraphrased sentences can break the text classifier.

# Try it now!

No matter how much experience you have on natural language processing and adversarial attack, we encourage you to try
the demo. Our demo is running on colab, **so you can try it without install!**

This colab will automatically download a sentiment classifier, and all required resources. When resources are
downloaded, you can type in your own sentences, and use Fibber to rewrite it. You can read the rewritten sentences, and
metric evaluation of rewritten sentence. You will see some rewritten sentences have the same meaning as your input but
get misclassified by the classifier.

**[Click here to Launch Colab!](https://colab.research.google.com/drive/1zefsU19P3HdrBUqJy7HU9b9cSaB_nBMP#scrollTo=uNcmhgzHJ3VQ)**


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

**Then you should install tensorflow and pytorch.** Please follow the instructions for [tensorflow](https://www.tensorflow.org/install) and [pytorch](https://pytorch.org). Fibber requires `tensorflow>=2.0.0` and `pytorch>=1.5.0`. Please choose a proper version of tensorflow and pytorch according to the CUDA version on your computer.


Remember to execute `conda activate fibber_env` every time you start a new console to work on **fibber**!

**Install Java** Please install a Java runtime environment on your computer.

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

**(2) Get a demo dataset and resources.**

```python
from fibber.datasets import get_demo_dataset

trainset, testset = get_demo_dataset()

from fibber.resources import download_all

# resources are downloaded to ~/.fibber
download_all()
```



**(3) Create a Fibber object.**

```python
from fibber.fibber import Fibber

# args starting with "bs_" are hyperparameters for the BertSamplingStrategy.
arg_dict = {
    "use_gpu_id": 0,
    "gpt2_gpu_id": 0,
    "bert_gpu_id": 0,
    "strategy_gpu_id": 0,
    "bs_block_size": 3,
    "bs_wpe_weight": 10000,
    "bs_use_weight": 1000,
    "bs_gpt2_weight": 10,
    "bs_clf_weight": 3
}

# create a fibber object.
# This step may take a while (about 1 hour) on RTX TITAN, and requires 20G of
# GPU memory. If there's not enough GPU memory on your GPU, consider assign use
# gpt2, bert, and strategy to different GPUs.
#
fibber = Fibber(arg_dict, dataset_name="demo", strategy_name="BertSamplingStrategy",
                trainset=trainset, testset=testset, output_dir="exp-demo")
```

**(4) You can also ask fibber to paraphrase your sentence.**

The following command can randomly paraphrase the sentence into 5 different ways.

```python
# Try sentences you like.
# label 0 means negative, and 1 means positive.
fibber.paraphrase(
    {"text0": ("The Avengers is a good movie. Although it is 3 hours long, every scene has something to watch."),
     "label": 1},
    field_name="text0",
    n=5)
```

The output is a tuple of (str, list, list).

```python
# Original Text
'The Avengers is a good movie. Although it is 3 hours long, every scene has something to watch.'

# 5 paraphrase_list
['the avengers is a good movie. even it is 2 hours long, there is not enough to watch.',
  'the avengers is a good movie. while it is 3 hours long, it is still very watchable.',
  'the avengers is a good movie and although it is 2 ¹⁄₂ hours long, it is never very interesting.',
  'avengers is not a good movie. while it is three hours long, it is still something to watch.',
  'the avengers is a bad movie. while it is three hours long, it is still something to watch.']

# Evaluation metrics of these 5 paraphrase_list.

  {'EditingDistance': 8,
   'USESemanticSimilarityMetric': 0.9523628950119019,
   'GloVeSemanticSimilarityMetric': 0.9795315341042675,
   'GPT2GrammarQualityMetric': 1.492070198059082,
   'BertClassifier': 0},
  {'EditingDistance': 9,
   'USESemanticSimilarityMetric': 0.9372092485427856,
   'GloVeSemanticSimilarityMetric': 0.9575780832312993,
   'GPT2GrammarQualityMetric': 0.9813404679298401,
   'BertClassifier': 1},
  {'EditingDistance': 11,
   'USESemanticSimilarityMetric': 0.9265919327735901,
   'GloVeSemanticSimilarityMetric': 0.9710499628056698,
   'GPT2GrammarQualityMetric': 1.325406551361084,
   'BertClassifier': 0},
  {'EditingDistance': 7,
   'USESemanticSimilarityMetric': 0.8913971185684204,
   'GloVeSemanticSimilarityMetric': 0.9800737898362042,
   'GPT2GrammarQualityMetric': 1.2504483461380005,
   'BertClassifier': 1},
  {'EditingDistance': 8,
   'USESemanticSimilarityMetric': 0.9124080538749695,
   'GloVeSemanticSimilarityMetric': 0.9744155151490856,
   'GPT2GrammarQualityMetric': 1.1626977920532227,
   'BertClassifier': 0}]
```

**(5) You can ask fibber to randomly pick a sentence from the dataset and paraphrase it.**


```python
fibber.paraphrase_a_random_sentence(n=5)
```



# Supported strategies

In this version, we implement three strategies

- IdentityStrategy:
	- The identity strategy outputs the original text as its paraphrase.
	- This strategy generates exactly 1 paraphrase for each original text regardless of `--num_paraphrases_per_text` flag.
- RandomStrategy:
	- The random strategy outputs the random shuffle of words in the original text.
- TextFoolerStrategy:
	- Implementation of [Jin et. al, 2019](https://arxiv.org/abs/1907.11932)
- BertSamplingStrategy:


# What's next?

For more details about **fibber** and all its possibilities
and features, please check the [documentation site](
https://DAI-Lab.github.io/fibber/).
