import logging
import os

import numpy as np
import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification

from . import data_utils, optim_utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_evaluate(model, dataloader_iter, eval_steps, summary, global_step):
  model.eval()

  correct, count = 0, 0
  loss_list = []

  for v_step in range(eval_steps):
    seq, mask, tok_type, label = next(dataloader_iter)
    seq = seq.to(DEVICE)
    mask = mask.to(DEVICE)
    tok_type = tok_type.to(DEVICE)
    label = label.to(DEVICE)

    with torch.no_grad():
      outputs = model(seq, mask, tok_type, labels=label)
      loss, logits = outputs[:2]
      loss_list.append(loss.detach().cpu().numpy())
      count += seq.size(0)
      correct += (logits.argmax(dim=1).eq(label)
                  .float().sum().detach().cpu().numpy())

  summary.add_scalar("clf_val/error_rate", 1 - correct / count, global_step)
  summary.add_scalar("clf_val/loss", np.mean(loss_list), global_step)

  model.train()


def save_model(model, opt, global_step, output_dir):
  states = {
      "model": model.state_dict(),
      "optimizer": opt.state_dict()
  }
  torch.save(states, output_dir
             + "/model/checkpoint-%04dk.pt" % (global_step // 1000))


def get_clf(FLAGS, trainset, testset):
  if trainset["cased"]:
    model_init = "bert-base-cased"
  else:
    model_init = "bert-base-uncased"
  num_labels = len(trainset["label_mapping"])

  model = BertForSequenceClassification.from_pretrained(
      model_init, num_labels=num_labels).to(DEVICE)
  model.train()

  logging.info("use %s tokenizer and classifier", model_init)
  logging.info("num labels: %s", num_labels)

  output_dir = FLAGS.output_dir + "/" + model_init

  ckpt_path = (output_dir + "/model/checkpoint-%04dk.pt" %
               (FLAGS.clf_step // 1000))

  if os.path.exists(ckpt_path):
    logging.info("load existing clf from %s", ckpt_path)
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict["model"])
    model.eval()
    return model

  summary = SummaryWriter(output_dir + "/summary")
  os.makedirs(output_dir + "/model", exist_ok=True)

  trainset = data_utils.Dataset(trainset, model_init, FLAGS.clf_bs)
  dataloader = torch.utils.data.DataLoader(
      trainset, batch_size=None, num_workers=2)

  testset = data_utils.Dataset(testset, model_init, FLAGS.clf_bs)
  dataloader_val = torch.utils.data.DataLoader(
      testset, batch_size=None, num_workers=1)
  dataloader_val_iter = iter(dataloader_val)

  params = model.parameters()

  opt, sche = optim_utils.get_optimizer(
      FLAGS.clf_opt, FLAGS.clf_lr, FLAGS.clf_decay, FLAGS.clf_step, params)

  global_step = 0
  correct_train, count_train = 0, 0
  for seq, mask, tok_type, label in tqdm.tqdm(dataloader, total=FLAGS.clf_step):
    global_step += 1
    seq = seq.to(DEVICE)
    mask = mask.to(DEVICE)
    tok_type = tok_type.to(DEVICE)
    label = label.to(DEVICE)

    outputs = model(seq, mask, tok_type, labels=label)
    loss, logits = outputs[:2]

    count_train += seq.size(0)
    correct_train += (logits.argmax(dim=1).eq(label)
                      .float().sum().detach().cpu().numpy())

    opt.zero_grad()
    loss.backward()
    opt.step()
    sche.step()

    if global_step % FLAGS.clf_period_summary == 0:
      summary.add_scalar("clf_train/loss", loss, global_step)
      summary.add_scalar("clf_train/error_rate", 1 - correct_train / count_train,
                         global_step)
      correct_train, count_train = 0, 0

    if global_step % FLAGS.clf_period_val == 0:
      run_evaluate(model, dataloader_val_iter,
                   FLAGS.clf_val_step, summary, global_step)

    if global_step % FLAGS.clf_period_save == 0:
      save_model(model, opt, global_step, output_dir)

    if global_step >= FLAGS.clf_step:
      break
  return model
