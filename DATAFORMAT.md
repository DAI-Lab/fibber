# Data Format

## Dataset format

Each dataset is stored in multiple JSON files. For example, the ag dataset is stored in `train.json` and `test.json`.

The JSON file contains the following fields:

- `label_mapping`: a list of strings. The `label_mapping` maps an integer label to the actual meaning of that label. This list is not used in the algorithm.
- `cased`: a bool value indicates if it is a cased dataset or uncased dataset. Sentences in uncased datasets are all in lowercase.
- `paraphrase_field`: choose from `text0` and `text1`. `Paraphrase_field` indicates which sentence in each data record should be paraphrased. 
- `data`: a list of data records. Each data records contains:
	- `label`: an integer indicating the classification label of the text.
	- `text0`:
		- For topic and sentiment classification datasets, text0 stores the text to be classified.
		- For natural language inference datasets, text0 stores the premise.
	- `text1`:
		- For topic and sentiment classification datasets, this field is omitted.
		- For natural language inference datasets, text1 stores the hypothesis.

A topic / sentiment classification example:

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

A natural langauge inference example:

```
{
  "label_mapping": [
    "neutral",
    "entailment",
    "contradiction"
  ],
  "cased": true,
  "paraphrase_field": "text1",
  "data": [
    {
      "label": 0,
      "text0": "A person on a horse jumps over a broken down airplane.",
      "text1": "A person is training his horse for a competition."
    },
    {
      "label": 2,
      "text0": "A person on a horse jumps over a broken down airplane.",
      "text1": "A person is at a diner, ordering an omelette."
    },
    ...
  ]
}
```


### Download datasets

We have scripts to help you easily download all datasets. We provide two options to download datasets:

**Download data preprocessed by us.** 

We preprocessed datasets and uploaded them to AWS. You can use the following command to download all datasets.

```
python3 -m fibber.datasets.download_datasets
```

After executing the command, the dataset is stored at `~/.fibber/datasets/<dataset_name>/*.json`. For example, the ag dataset is stored in `~/.fibber/datasets/ag/`. And there will be two sets `train.json` and `test.json` in the folder.

**Download and process data from the original source.** 

You can also download the original dataset version and process it locally.

```
python3 -m fibber.datasets.download_datasets --process_raw 1
```
This script will download data from the original source to `~/.fibber/datasets/<dataset_name>/raw/` folder. And process the raw data to generate the JSON files.


## Output format

During the benchmark process, we save results in several files.

### Intermediate result

The intermediate result `<output_dir>/<dataset>-<strategy>-<date>-<time>-tmp.json` stores the paraphrased sentences. Strategies can run for a few minutes (hours) on some datasets, so we save the result every 30 seconds. The file format is similar to the dataset file. For each data record, we add a new field, `text0_paraphrases` or `text1_paraphrases` depending the `paraphrase_field`.

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
      "text0_paraphrases": ["The 2008 NBA championship is won by Boston.", ...]
    },
    ...
  ]
}
```

### Result with metrics

The result `<output_dir>/<dataset>-<strategy>-<date>-<time>-with-metrics.json` stores the paraphrased sentences as well as metrics. Compute metrics may need a few minutes on some datasets, so we save the result every 30 seconds. The file format is similar to the intermediate file. For each data record, we add two new field, `original_text_metrics` and `paraphrase_metrics`.

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
      "original_text_metrics": {
        "EditDistanceMetric": 0,
        "USESemanticSimilarityMetric": 1.0,
        "GloVeSemanticSimilarityMetric": 1.0,
        "GPT2GrammarQualityMetric": 1.0,
        "BertClassifier": 1
      },
      "paraphrase_metrics": [
        {
          "EditDistanceMetric": 7,
          "USESemanticSimilarityMetric": 0.91,
          "GloVeSemanticSimilarityMetric": 0.94,
          "GPT2GrammarQualityMetric": 2.3,
          "BertClassifier": 1
        },
        ...
      ]
    },
    ...
  ]
}
```

The `original_text_metrics` stores a dict of several metrics. It compares the original text against itself. The `paraphrase_metrics` is a list of the same length as paraphrases in this data record. Each element in this list is a dict showing the comparison between the original text and one paraphrased text.
