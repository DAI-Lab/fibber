import torch
import transformers

def get_optimizer(optim, lr, decay, train_step, params, warmup=500):
  if optim == "adam":
    opt = torch.optim.Adam(params=params, lr=lr, weight_decay=decay)
  elif optim == "sgd":
    opt = torch.optim.SGD(params=params, lr=lr, weight_decay=decay)
  elif optim == "adamw":
    opt = torch.optim.AdamW(params=params, lr=lr, weight_decay=decay)
  else:
    assert 0, "unkown optimizer"

  sche = transformers.get_linear_schedule_with_warmup(
      opt, num_warmup_steps=1000, num_training_steps=train_step)

  return opt, sche
