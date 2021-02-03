# History

## Version 0.2.2 - 2021-02-3
This release fixes bugs and adds unit tests.

### New Features

- Add Sentence BERT metric and corresponding unit test.
- Fix the bug of the colab demo.


## Version 0.2.1 - 2021-01-20
This release improves documentation and unit tests.

### New Features

- Add integrity test for IdentityStrategy, TextAttackStrategy, and BertSamplingStrategy.
- For IdentityStrategy and TextAttackStrategy, accuracy is verified.
- Improve documentation, split data format from benchmark.


## Version 0.2.0 - 2021-01-06
This release updates the structure of the project and improve documentation.

### New Features

- Metric module is redesigned to have a consistant API. ([Issue #12](https://github.com/DAI-Lab/fibber/issues/12))
- More unit tests are added. Slow unit tests are skipped in CI. ([Issue #11](https://github.com/DAI-Lab/fibber/issues/11))
- Benchmark table is updated. ([Issue #10](https://github.com/DAI-Lab/fibber/issues/10))
- Better support to `TextAttack`. Users can choose any implemented attacking method in `TextAttack` using the `ta_recipe` arg. ([Issue #9](https://github.com/DAI-Lab/fibber/issues/9))


## Version 0.1.3

This release includes the following updates:

- Add a benchmark class. Users can integrate fibber benchmark to other projects. The class supports customized datasets, target classifier and attacking method.
- Migrate from Travis CI to Github Action.
- Move adversarial-attack-related aggragation functions from benchmark module to metric module.

## Version 0.1.2

This minor release add pretrained classifiers and downloadable resources on a demo dataset, and a
demo Colab.

## Version 0.1.1

This minor release removes the dependency on `textattack` because it produces dependency conflicts.
Users can install it manually to use attacking strategies in `textattack`.

## version 0.1.0

This release is a major update to Fibber library. Advanced paraphrase algorithms are included.

- Add two strategies: TextFoolerStrategy and BertSamplingStrategy.
- Improve the benchmarking framework: add more metrics specifically designed for adversarial attack.
- Datasets: add a variation of AG's news dataset, `ag_no_title`.
- Bug fix and improvements.

## version 0.0.1

This is the first release of Fibber library. This release contains:

- Datasets: fibber contains 6 built-in datasets.
- Metrics: fibber contains 6 metrics to evaluate the quality of paraphrased
  sentences. All metrics have a unified interface.
- Benchmark framework: the benchmark framework and easily evaluate the
  phraphrase strategies on built-in datasets and metrics.
- Strategies: this release contains 2 basic strategies, the identity strategy
  and random strategy.
- A unified Fibber interface: users can easily use fibber by creating a Fibber
  object.
