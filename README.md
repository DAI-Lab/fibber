<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/fibber.svg)](https://pypi.python.org/pypi/fibber)-->
<!--[![Downloads](https://pepy.tech/badge/fibber)](https://pepy.tech/project/fibber)-->
[![Travis CI Shield](https://travis-ci.com/DAI-Lab/fibber.svg?token=g6BnJQz9Aaqdj1paqcNM&branch=master&status=started)](https://travis-ci.org/DAI-Lab/fibber)
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
python -m fibber.pipeline.download_datasets
python -m fibber.pipeline.benchmark
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

**(2) Download datasets**

Please use the following command to download all datasets.

```bash
python -m fibber.pipeline.download_datasets
```

All datasets will be downloaded and stored at `~/.fibber/datasets`.

**(3) Execute the benchmark on one dataset using one paraphrase strategy.**

The following command will run the `random` strategy on the `ag` dataset. To use other datasets, see the [datasets](#Datasets) section.

```bash
python -m fibber.pipeline.benchmark \
	--dataset ag \
	--strategy RandomStrategy \
	--output_dir exp-ag \
	--num_paraphrases_per_text 20 \
	--subsample_testset 100 \
	--gpt2_gpu 0 \
	--bert_gpu 0 \
	--use_gpu 0 \
	--bert_clf_steps 20000
```

It first subsamples the test set to `100` examples, then generates `20` paraphrases for each example. During this process, the paraphrased sentences will be stored at `exp-ag/ag-RandomStrategy-<date>-<time>-tmp.json`.

Then the pipeline will initialize all the evaluation metrics.

- We will use a `GPT2` model to evaluate if a sentence is meaningful. The `GPT2` language model will be executed on `gpt2_gpu`. You should change the argument to a proper GPU id.
- We will use a `Universal sentence encoder (USE)` model to measure the similarity between two paraphrased sentences and the original sentence. The `USE` will be executed on `use_gpu`. You should change the argument to a proper GPU id.
- We will use a `BERT` model to predict the classification label for paraphrases. The `BERT` will be executed on `bert_gpu`. You should change the argument to a proper GPU id. **Note that the BERT classifier will be trained for the first time you execute the pipeline. Then the trained model will be saved at `~/.fibber/bert_clf/<dataset_name>/`. Because of the training, it will use more GPU memory than GPT2 and USE. So assign BERT to a separate GPU if you have multiple GPUs.**

After the execution, the evaluation metric for each of the paraphrases will be stored at `exp-ag/ag-RandomStrategy-<date>-<time>-with-measurement.json`.

The aggregated result will be stored as a row at `~/.fibber/results/detailed.csv`.

**(4) Generate overview result.**

We use the number of wins to compare different strategies. To generate the overview table, use the following command. 

```bash
python -m fibber.pipeline.make_overview
```

The overview table will be stored at `~/.fibber/results/overview.csv`.

Before running this command, please verify `~/.fibber/results/detailed.csv`. Each strategy must not have more than one executions on one dataset. Otherwise, the script will raise assertion errors. 


# Benchmark result

The following table shows the benchmarking result. (Here we show the number of wins.)

| model name       | Paraphrased Text Accuracy (USE\_sim > 0.95, GPT2\_score<2 | Paraphrased Text Accuracy (USE\_sim > 0.90, GPT2\_score<5 | USESemanticSimilarity mean | GPT2GrammarQuality mean |
|--------------------|------------------------------|------------------------------|----------------------------|-------------------------|
| IdenticalStrategy  | 0                            | 0                            | 14                         | 14                      |
| RandomStrategy     | 1                            | 5                            | 2                          | 0                       |
| TextFoolerStrategy | 14                           | 14                           | 5                          | 7                       |

For detailed tables, see [Google Sheet](https://docs.google.com/spreadsheets/d/1B_5RiMfndNVhxZLX5ykMqt5SCjpy3MxOovBi_RL41Fw/edit?usp=sharing).

# Specifications

## Datasets

Here is the information about datasets in fibber.

| Type                       | Name                    | Size (train/test) | Classes                             |
|----------------------------|-------------------------|-------------------|-------------------------------------|
| Topic Classification       | [ag](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)| 120k / 7.6k       | World / Sport / Business / Sci-Tech                                                                                |
| Sentiment classification   | [mr](http://www.cs.cornell.edu/people/pabo/movie-review-data/)           | 9k / 1k           |  Negative / Positive |
| Sentiment classification   | [yelp](https://academictorrents.com/details/66ab083bda0c508de6c641baabb1ec17f72dc480) | 160k / 38k        | Negative / Positive                 |
| Sentiment classification   | [imdb](https://ai.stanford.edu/~amaas/data/sentiment/)| 25k / 25k         | Negative / Positive                 |
| Natural Language Inference | [snli](https://nlp.stanford.edu/projects/snli/) | 570k / 10k        | Entailment / Neutral / Contradict   |
| Natural Language Inference | [mnli](https://cims.nyu.edu/~sbowman/multinli/)                | 433k / 10k        | Entailment / Neutral / Contradict   |

Note that mnli has two configurations. Use `mnli` for matched testset, and `mnli_mis` for mismatched testset.


### Dataset format

Each dataset is stored in multiple JSON files. For example, the ag dataset is stored in `train.json` and `test.json`.

The JSON file contains the following fields:

- label\_mapping: a list of strings. The label_mapping maps an integer label to the actual meaning of that label. This list is not used in the algorithm.
- cased: a bool value indicates if it is a cased dataset or uncased dataset. Sentences in uncased datasets are all in lowercase.
paraphrase\_field: choose from text0 and text1. Paraphrase_field indicates which sentence in each data record should be paraphrased.
- data: a list of data records. Each data records contains:
	- label: an integer indicating the classification label of the text.
	- text0:
		- For topic and sentiment classification datasets, text0 stores the text to be classified.
		- For natural language inference datasets, text0 stores the premise.
	- text1:
		- For topic and sentiment classification datasets, this field is omitted.
		- For natural language inference datasets, text1 stores the hypothesis.

Here is an example:

```
{
  "label_mapping": [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
  ],
  "cased": true,
  "paraphrase_field": "text0",
  "data": [
    {
      "label": 1,
      "text0": "Boston won the NBA championship in 2008."
    },
    {
      "label": 3,
      "text0": "Apple releases its latest cell phone."
    },
    ...
  ]
}
```

### Download datasets

We have scripts to help you easily download all datasets. We provide two options to download datasets:

- **Download data preprocessed by us.** We preprocessed datasets and uploaded them to AWS. You can use the following command to download all datasets.
```
python3 -m fibber.pipeline download_datasets
```
After executing the command, the dataset is stored at `~/.fibber/datasets/<dataset_name>/*.json`. For example, the ag dataset is stored in `~/.fibber/datasets/ag/`. And there will be two sets `train.json` and `test.json` in the folder.
- **Download and process data from the original source.** You can also download the original dataset version and process it locally.
```
python3 -m fibber.pipeline download_datasets --process_raw 1
```
This script will download data from the original source to `~/.fibber/datasets/<dataset_name>/raw/` folder. And process the raw data to generate the JSON files.

## Supported strategies

In this version, we implement three strategies

- IdenticalStrategy: 
	- The identical strategy outputs the original text as its paraphrase. 
	- This strategy generates exactly 1 paraphrase for each original text regardless of `--num_paraphrases_per_text` flag.
- RandomStrategy:
	- The random strategy outputs the random shuffle of words in the original text.
- TextFoolerStrategy:
	- The TextFooler strategy uses TextFooler to attack the text classifier, if the attack succeeds, outputs the adversarial text, otherwise outputs the original text.
	- This strategy generates exactly 1 paraphrase for each original text regardless of `--num_paraphrases_per_text` flag.


## Output format

During the benchmark process, we save results in several files.

### Intermediate result

The intermediate result `<output_dir>/<dataset>-<strategy>-<date>-<time>-tmp.json` stores the paraphrased sentences. Strategies can run for a few minutes (hours) on some datasets, so we save the result every 30 seconds. The file format is similar to the dataset file. For each data record, we add a new field, `text0_paraphrases` or `text1_paraphrases` depending o the `paraphrase_field`.

An example is as follows.

```
{
  "label_mapping": [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
  ],
  "cased": true,
  "paraphrase_field": "text0",
  "data": [
    {
      "label": 1,
      "text0": "Boston won the NBA championship in 2008.",
      "text0_paraphrases": [..., ...]
    },
    ...
  ]
}
```

### Result with measurement

The result `<output_dir>/<dataset>-<strategy>-<date>-<time>-with-measurement.json` stores the paraphrased sentences as well as measurements. Measurements can run for a few minutes on some datasets, so we save the result every 30 seconds. The file format is similar to the intermediate file. For each data record, we add two new field, `original_text_measurements` and `paraphrase_measurements`. 

An example is as follows.

```
{
  "label_mapping": [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
  ],
  "cased": true,
  "paraphrase_field": "text0",
  "data": [
    {
      "label": 1,
      "text0": "Boston won the NBA championship in 2008.",
      "text0_paraphrases": [..., ...],
      "original_text_measurements": {
        "EditingDistance": 0,
        "USESemanticSimilarity": 1.0,
        "GloVeSemanticSimilarity": 1.0,
        "GPT2GrammarQuality": 1.0,
        "BertClfPrediction": 1
      },
      "paraphrase_measurements": [
        {
          "EditingDistance": 7,
          "USESemanticSimilarity": 0.91,
          "GloVeSemanticSimilarity": 0.94,
          "GPT2GrammarQuality": 2.3,
          "BertClfPrediction": 1
        }, 
        ...
      ]
    },
    ...
  ]
}
```

The `original_text_measurements` stores a dictionary of several metrics. It compares the original text against itself. The `paraphrase_measurements` is a list of the same length as paraphrases in this data record. Each element in this list is a dictionary showing the comparison between the original text and one paraphrased text.



# What's next?

For more details about **fibber** and all its possibilities
and features, please check the [documentation site](
https://DAI-Lab.github.io/fibber/).
