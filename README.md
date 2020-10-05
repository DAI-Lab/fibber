<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/fibber.svg)](https://pypi.python.org/pypi/fibber)-->
<!--[![Downloads](https://pepy.tech/badge/fibber)](https://pepy.tech/project/fibber)-->
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/fibber.svg?branch=master)](https://travis-ci.org/DAI-Lab/fibber)
[![Coverage Status](https://codecov.io/gh/DAI-Lab/fibber/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/fibber)



# fibber

Fibber is a benchmarking suite for adversarial attacks on text classification.

- Documentation: https://DAI-Lab.github.io/fibber
- Homepage: https://github.com/DAI-Lab/fibber

# Overview

TODO: Provide a short overview of the project here.

# Datasets
Here are the information of datasets in fibber.

| Type                       | Name                    | Size (train/test) | Classes                             |
|----------------------------|-------------------------|-------------------|-------------------------------------|
| Topic Classification       | [ag](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)| 120k / 7.6k       | World / Sport / Business / Sci-Tech                                                                                |
| Sentiment classification   | [mr](http://www.cs.cornell.edu/people/pabo/movie-review-data/)           | 9k / 1k           |  Negative / Positive |
| Sentiment classification   | [yelp](https://academictorrents.com/details/66ab083bda0c508de6c641baabb1ec17f72dc480) | 160k / 38k        | Negative / Positive                 | 
| Sentiment classification   | [imdb](https://ai.stanford.edu/~amaas/data/sentiment/)| 25k / 25k         | Negative / Positive                 |
| Natural Language Inference | [snli](https://nlp.stanford.edu/projects/snli/) | 570k / 10k        | Entailment / Neutral / Contradict   |                                                                                                            |
| Natural Language Inference | [mnli](https://cims.nyu.edu/~sbowman/multinli/)                | 433k / 10k        | Entailment / Neutral / Contradict   | 


## Format
Each dataset is stored in multiple json files. For example, the ag dataset is stored in `train.json` and `test.json`.

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

## Download datasets
We have scripts to help you easily download all datasets. We provide two options to download datasets:

- **Download data preprocessed by us.** We preprocessed datasets and uploaded to aws. You can use the following command to download all datasets. 
```
python3 -m fibber.pipeline download_datasets
```
After executing the command, the dataset is stored at `~/.fibber/datasets/<dataset_name>/*.json`. For example, the ag dataset is stored in `~/.fibber/datasets/ag/`. And there will be two sets `train.json` and `test.json` in the folder.
- **Download and process data from original source.** You can also download the orginal dataset version and process it locally. 
```
python3 -m fibber.pipeline download_datasets --process_raw 1
```
This script will download data from the original source to `~/.fibber/datasets/<dataset_name>/raw/` folder. And process the raw data to generate the json files. 




# Install

## Requirements

**fibber** has been developed and tested on [Python 3.5, 3.6, 3.7 and 3.8](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **fibber** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **fibber**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) fibber-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source fibber-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **fibber**!

<!-- Uncomment this section after releasing the package to PyPI for installation instructions
## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **fibber**:

```bash
pip install fibber
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/fibber.git
cd fibber
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://DAI-Lab.github.io/fibber/contributing.html#get-started)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **fibber**.

TODO: Create a step by step guide here.

# What's next?

For more details about **fibber** and all its possibilities
and features, please check the [documentation site](
https://DAI-Lab.github.io/fibber/).
