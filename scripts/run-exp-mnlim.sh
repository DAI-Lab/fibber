python3 -m fibber.benchmark.benchmark --output_dir exp-mnli_mis --dataset mnli_mis --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --subsample_testset 500 --strategy IdentityStrategy

python3 -m fibber.benchmark.benchmark --output_dir exp-mnli_mis --dataset mnli_mis --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --subsample_testset 500 --strategy RandomStrategy

python3 -m fibber.benchmark.benchmark --num_paraphrases_per_text 50 \
  --output_dir exp-mnli_mis --dataset mnli_mis --use_gpu_id 0 --bert_gpu_id 0 --gpt2_gpu_id 0 --strategy_gpu_id 0 \
  --subsample_testset 500 --bert_clf_steps 20000 --strategy BertSamplingStrategy \
  --bs_enforcing_dist wpe --bs_wpe_threshold 1.0 --bs_wpe_weight 1000 --bs_burnin_enforcing_schedule 1 \
  --bs_accept_criteria joint_weighted_criteria --bs_burnin_criteria_schedule 1 --bs_use_threshold 0.95 --bs_use_weight 200 \
  --bs_seed_option origin --bs_split_sentence auto --bs_sampling_steps 200 --bs_burnin_steps 100 \
  --bs_lm_option finetune --bs_clf_weight 5 --bs_stanza_port 9001 --bs_window_size 3 --bs_gpt2_weight 5
