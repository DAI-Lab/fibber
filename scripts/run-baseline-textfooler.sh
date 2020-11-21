# TextFooler Strategy
python3 -m fibber.benchmark.benchmark --output_dir exp-ag_no_title --dataset ag_no_title --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --subsample_testset 500 --strategy TextFoolerStrategy
python3 -m fibber.benchmark.benchmark --output_dir exp-mr --dataset mr --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --subsample_testset 500 --bert_clf_steps 5000 --strategy TextFoolerStrategy
python3 -m fibber.benchmark.benchmark --output_dir exp-yelp --dataset yelp --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --subsample_testset 500 --strategy TextFoolerStrategy
python3 -m fibber.benchmark.benchmark --output_dir exp-imdb --dataset imdb --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --subsample_testset 500 --bert_clf_steps 5000 --strategy TextFoolerStrategy
python3 -m fibber.benchmark.benchmark --output_dir exp-snli --dataset snli --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --subsample_testset 500 --strategy TextFoolerStrategy
python3 -m fibber.benchmark.benchmark --output_dir exp-mnli --dataset mnli --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --subsample_testset 500 --strategy TextFoolerStrategy
python3 -m fibber.benchmark.benchmark --output_dir exp-mnli_mis --dataset mnli_mis --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --subsample_testset 500 --strategy TextFoolerStrategy
