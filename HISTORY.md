# History

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
