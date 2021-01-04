# Benchmark

Benchmark module is an important component in Fibber. It provides an easy-to-use
API and is highly customizable. In this document, we will show

- Benchmark result: we benchmark all builtin methods on builtin dataset.
- Basic usage: how to use builtin strategies to attack BERT classifier on a builtin dataset.
- Advance usage: how to customize strategy, classifier, and dataset.

## Benchmark result

The following table shows the benchmarking result. (Here we show the number of wins.)

| paraphrase\_strategy\_name | BertClas\_sifier\_AfterAttackAccuracy | BertClassifier\_best\_sim\_adv\_EditDistanceMetric | BertClassifier\_best\_sim\_adv\_GPT2GrammarQualityMetric | BertClassifier\_best\_sim\_adv\_GloVeSemanticSimilarityMetric | BertClassifier\_best\_sim\_adv\_USESemanticSimilarityMetric | EditDistanceMetric | GPT2GrammarQualityMetric | GloVeSemanticSimilarityMetric | PairwiseEditDistance | USESemanticSimilarityMetric |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
|--------------------------|------------------------------------|------------------------------------------------|------------------------------------------------------|-----------------------------------------------------------|---------------------------------------------------------|--------------------|--------------------------|-------------------------------|----------------------|-----------------------------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IdentityStrategy         | 0                                  | 0                                              | 0                                                    | 0                                                         | 0                                                       | 0                  | 32                       | 33                            | 0                    | 33                          |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| RandomStrategy           | 5                                  | 16                                             | 0                                                    | 16                                                        | 6                                                       | 33                 | 0                        | 26                            | 7                    | 10                          |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| TextFoolerJin2019        | 25                                 | 8                                              | 12                                                   | 6                                                         | 11                                                      | 19                 | 9                        | 5                             | 0                    | 9                           |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| BAEGarg2019              | 13                                 | 5                                              | 5                                                    | 14                                                        | 13                                                      | 10                 | 13                       | 8                             | 0                    | 13                          |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| PSOZang2020              | 16                                 | 3                                              | 12                                                   | 6                                                         | 5                                                       | 9                  | 15                       | 9                             | 0                    | 8                           |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| BertSamplingStrategy     | 33                                 | 19                                             | 23                                                   | 10                                                        | 17                                                      | 24                 | 26                       | 14                            | 0                    | 22                          |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
For detailed tables, see [Google Sheet](https://docs.google.com/spreadsheets/d/1B_5RiMfndNVhxZLX5ykMqt5SCjpy3MxOovBi_RL41Fw/edit?usp=sharing).


## Basic Usage

In this short tutorial, we will guide you through a series of steps that will help you run
benchmark on builtin strategies and datasets.

### Preparation

**Install Fibber:** Please follow the instructions to [Install Fibber](#Install).**

**Download datasets:** Please use the following command to download all datasets.

```bash
python -m fibber.datasets.download_datasets
```

All datasets will be downloaded and stored at `~/.fibber/datasets`.

### Run benchmark as a module

If you are trying to reproduce the performance table, running the benchmark as a module is
recommended.

The following command will run the `BertSamplingStrategy` strategy on the `mr` dataset. To use other
datasets, see the [datasets](#Datasets) section.

```bash
python -m fibber.benchmark.benchmark \
	--dataset mr \
	--strategy BertSamplingStrategy \
	--output_dir exp-mr \
	--num_paraphrases_per_text 20 \
	--subsample_testset 100 \
	--gpt2_gpu 0 \
	--bert_gpu 0 \
	--use_gpu 0 \
	--bert_clf_steps 20000
```

It first subsamples the test set to `100` examples, then generates `20` paraphrases for each
example. During this process, the paraphrased sentences will be stored at
`exp-mr/mr-BertSamplingStrategy-<date>-<time>-tmp.json`.

Then the pipeline will initialize all the evaluation metrics.

- We will use a `GPT2` model to evaluate if a sentence is meaningful. The `GPT2` language model will be executed on `gpt2_gpu`. You should change the argument to a proper GPU id.
- We will use a `Universal sentence encoder (USE)` model to measure the similarity between two paraphrased sentences and the original sentence. The `USE` will be executed on `use_gpu`. You should change the argument to a proper GPU id.
- We will use a `BERT` model to predict the classification label for paraphrases. The `BERT` will be executed on `bert_gpu`. You should change the argument to a proper GPU id. **Note that the BERT classifier will be trained for the first time you execute the pipeline. Then the trained model will be saved at `~/.fibber/bert_clf/<dataset_name>/`. Because of the training, it will use more GPU memory than GPT2 and USE. So assign BERT to a separate GPU if you have multiple GPUs.**

After the execution, the evaluation metric for each of the paraphrases will be stored at `exp-ag/ag-RandomStrategy-<date>-<time>-with-metrics.json`.

The aggregated result will be stored as a row at `~/.fibber/results/detailed.csv`.

### Run in a python script / jupyter notebook

You may want to integrate the benchmark framework into your own python script. We also provide easy to use APIs.

**Create a Benchmark object** The following code will create a fibber Benchmark object on `mr` dataset.

```
from fibber.benchmark import Benchmark

benchmark = Benchmark(
    output_dir = "exp-debug",
    dataset_name = "mr",
    subsample_attack_set=100,
    use_gpu_id=0,
    gpt2_gpu_id=0,
    bert_gpu_id=0,
    bert_clf_steps=1000,
    bert_clf_bs=32
)
```

Similarly, you can assign different components to different GPUs.

**Run benchmark** Use the following code to run the benchmark using a specific strategy.  

```
benchmark.run_benchmark(paraphrase_strategy="BertSamplingStrategy")
```

### Generate overview result

We use the number of wins to compare different strategies. To generate the overview table, use the following command.

```bash
python -m fibber.benchmark.make_overview
```

The overview table will be stored at `~/.fibber/results/overview.csv`.

Before running this command, please verify `~/.fibber/results/detailed.csv`. Each strategy must not have more than one executions on one dataset. Otherwise, the script will raise assertion errors.


## Advanced Usage

### Customize dataset

To run a benchmark on a customized classification dataset, you should first convert a dataset into fibber's standard data format.

Then construct a benchmark object using your own dataset.

```
benchmark = Benchmark(
    output_dir = "exp-debug",
    dataset_name = "customized_dataset",
		trainset = your_train_set,
		testset = your_test_set,
		attack_set = your_attack_set,
    subsample_attack_set=0,
    use_gpu_id=0,
    gpt2_gpu_id=0,
    bert_gpu_id=0,
    bert_clf_steps=1000,
    bert_clf_bs=32
)
```

### Customize classifier

To customize classifier, use the `customized_clf` arg in Benchmark. For example,

```
# a naive classifier that always outputs 0.
class CustomizedClf(MetricBase):
	def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
		return 0

benchmark = Benchmark(
    output_dir = "exp-debug",
    dataset_name = "mr",
		customized_clf=CustomizedClf(),
    subsample_attack_set=0,
    use_gpu_id=0,
    gpt2_gpu_id=0,
    bert_gpu_id=0,
    bert_clf_steps=1000,
    bert_clf_bs=32
)
```

### Customize strategy

To customize strategy, you should create a strategy object then call the `run_benchmark` function. For example,
we want to benchmark BertSamplingStrategy using a different set of hyper parameters.

```
strategy = BertSamplingStrategy(
    arg_dict={"bs_clf_weight": 0},
		dataset_name="mr",
		strategy_gpu_id=0,
		output_dir="exp_mr",
		metric_bundle=benchmark.get_metric_bundle())

benchmark.run_benchmark(strategy)
```

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

Note that ag has two configurations. In `ag`, we combines the title and content as input for classification. In `ag_no_title`, we use only use content as input.

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
python3 -m fibber.datasets.download_datasets
```
After executing the command, the dataset is stored at `~/.fibber/datasets/<dataset_name>/*.json`. For example, the ag dataset is stored in `~/.fibber/datasets/ag/`. And there will be two sets `train.json` and `test.json` in the folder.
- **Download and process data from the original source.** You can also download the original dataset version and process it locally.
```
python3 -m fibber.datasets.download_datasets --process_raw 1
```
This script will download data from the original source to `~/.fibber/datasets/<dataset_name>/raw/` folder. And process the raw data to generate the JSON files.


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
        "EditingDistance": 0,
        "USESemanticSimilarity": 1.0,
        "GloVeSemanticSimilarity": 1.0,
        "GPT2GrammarQuality": 1.0,
        "BertClfPrediction": 1
      },
      "paraphrase_metrics": [
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

The `original_text_metrics` stores a dict of several metrics. It compares the original text against itself. The `paraphrase_metrics` is a list of the same length as paraphrases in this data record. Each element in this list is a dict showing the comparison between the original text and one paraphrased text.
