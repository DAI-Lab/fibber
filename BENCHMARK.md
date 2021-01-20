# Benchmark

Benchmark module is an important component in Fibber. It provides an easy-to-use
API and is highly customizable. In this document, we will show

- Built-in Datasets: we preprocessed 6 datasets into [fibber's format](https://dai-lab.github.io/fibber/dataformat.html).
- Benchmark result: we benchmark all built-in methods on built-in dataset.
- Basic usage: how to use builtin strategies to attack BERT classifier on a built-in dataset.
- Advance usage: how to customize strategy, classifier, and dataset.

## Built-in Datasets

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

## Benchmark result

The following table shows the benchmarking result. (Here we show the number of wins.)

| StrategyName         | AfterAttackAccuracy | GPT2GrammarQuality | GloVeSemanticSimilarity | USESemanticSimilarity |
|----------------------|---------------------|--------------------|-------------------------|-----------------------|
| IdentityStrategy     | 0                   | 0                  | 0                       | 0                     |
| RandomStrategy       | 5                   | 0                  | 16                      | 6                     |
| TextFoolerJin2019    | 25                  | 12                 | 6                       | 11                    |
| BAEGarg2019          | 13                  | 5                  | 14                      | 13                    |
| PSOZang2020          | 16                  | 12                 | 6                       | 5                     |
| BertSamplingStrategy | 33                  | 23                 | 10                      | 17                    |

For detailed tables, see [Google Sheet](https://docs.google.com/spreadsheets/d/1B_5RiMfndNVhxZLX5ykMqt5SCjpy3MxOovBi_RL41Fw/edit?usp=sharing).


## Basic Usage

In this short tutorial, we will guide you through a series of steps that will help you run
benchmark on builtin strategies and datasets.

### Preparation

**Install Fibber:** Please follow the instructions to [Install Fibber](https://dai-lab.github.io/fibber/readme.html#install).**

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

To run a benchmark on a customized classification dataset, you should first convert a dataset into [fibber's format](https://dai-lab.github.io/fibber/dataformat.html).

Then construct a benchmark object using your own dataset.

```
benchmark = Benchmark(
    output_dir = "exp-debug",
    dataset_name = "customized_dataset",
    
    ### Pass your processed datasets here. ####
    trainset = your_train_set,
    testset = your_test_set,
    attack_set = your_attack_set,
    ###########################################
    
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
class CustomizedClf(ClassifierBase):
	def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
		return 0

benchmark = Benchmark(
    output_dir = "exp-debug",
    dataset_name = "mr",
    
    # Pass your customized classifier here. 
    # Note that the Benchmark class will NOT train the classifier.
    # So please train your classifier before pass it to Benchmark.
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

