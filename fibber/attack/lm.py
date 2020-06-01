import logging
import os

import numpy as np
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForMaskedLM

from .. import data_utils, optim_utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def construct_lm_text_example(tokenizer, seq, mask, label, lm_label, lm_pred):
  classes = ["World", "Sports", "Business", "Sci/Tech"]

  ret = "[%s] " % classes[label]

  for x, m, y, z in zip(seq, mask, lm_label, lm_pred):
    if m == 0:
      break
    if y == -100:
      ret += " %s" % escape(tokenizer.convert_ids_to_tokens([x])[0])
      continue

    y = y % tokenizer.vocab_size
    z = z % tokenizer.vocab_size

    x, y, z = tokenizer.convert_ids_to_tokens([x, y, z])
    if x != tokenizer.mask_token:
      ret += " __%s" % escape(x)
    else:
      ret += " __"

    if y == z:
      # correct
      ret += "=%s__" % escape(y)
    else:
      # incorrect
      ret += "=%s=%s__" % (escape(y), escape(z))

  return ret


def write_summary(lm_model, tokenizer, stats, summary, global_step,
                  is_summary_iter, example_ids=[0, 5, 10]):
  summary.add_scalar("attack/loss_lm", np.mean(stats["lm_loss"]), global_step)
  summary.add_scalar("attack/error_lm",
                     1 - stats["lm_correct"] / stats["lm_total"], global_step)

  # example_seq
  # example_mask
  # example_label
  # example_lm_label
  # example_lm_pred

  if is_summary_iter:
    text = ""
    for index in example_ids:
      text += construct_lm_text_example(
          tokenizer,
          stats["example_seq"][index],
          stats["example_mask"][index],
          stats["example_label"][index],
          stats["example_lm_label"][index],
          stats["example_lm_pred"][index]) + "\n\n"

    summary.add_text("log", text, global_step)


def save_model(lm_model, opt, global_step, output_dir):
  states = {
      "lm_model": lm_model.state_dict(),
      "opt": opt.state_dict()
  }
  torch.save(states, output_dir
             + "/model/checkpoint-%04dk.pt" % (global_step // 1000))


def new_stats():
  return {
      "lm_total": 0,
      "lm_correct": 0,
      "lm_loss": [],
  }


def compute_lm_loss(lm_model, seq, mask, tok_type,
                    lm_label, stats, is_summary_iter):
  lm_hid = lm_model.bert(seq, mask, tok_type)[0]
  lm_hid = torch.masked_select(lm_hid, lm_label.gt(
      0).unsqueeze(2)).view(-1, lm_hid.size(2))
  logits = lm_model.cls(lm_hid)

  with torch.no_grad():
    lm_label_squeeze = torch.masked_select(lm_label, lm_label.gt(0))

  lm_loss = F.cross_entropy(logits, lm_label_squeeze)

  if is_summary_iter:
    with torch.no_grad():
      tmp = torch.zeros_like(seq)
      tmp[lm_label.gt(0)] = logits.argmax(dim=1)
      stats["example_lm_pred"] = tmp.detach().cpu().numpy()

  # lm_label is -100 for unmasked token or padding toks.
  stats["lm_total"] += (lm_label_squeeze > 0).int().sum().detach().cpu().numpy()
  stats["lm_correct"] += (logits.argmax(dim=1).eq(lm_label_squeeze)
                          .float().sum().detach().cpu().numpy())
  stats["lm_loss"].append(lm_loss.detach().cpu().numpy())

  return lm_loss


def prepare_lm(FLAGS, trainset, filter):
  if filter == -1:
    output_dir = FLAGS.output_dir + "/lm_all"
  else:
    output_dir = FLAGS.output_dir + "/lm_no_c%d" % filter

  summary = SummaryWriter(output_dir + "/summary")
  os.makedirs(output_dir + "/model", exist_ok=True)

  if trainset["cased"]:
    model_init = "bert-base-cased"
  else:
    model_init = "bert-base-uncased"

  ckpt_path = (output_dir + "/model/checkpoint-%04dk.pt" %
               (FLAGS.lm_step // 1000))
  if os.path.exists(ckpt_path):
    logging.info("Lm exists %s", ckpt_path)
    return

  lm_model = BertForMaskedLM.from_pretrained(model_init).to(DEVICE)
  lm_model.train()

  dataset = data_utils.Dataset(trainset, model_init, FLAGS.lm_bs,
                               exclude=filter, masked_lm=True)
  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=None, num_workers=2)

  if filter != -1:
    ckpt_path = (FLAGS.output_dir + "/lm_all/model/checkpoint-%04dk.pt" %
                 (FLAGS.lm_step // 1000))
    logging.info("load general lm from %s", ckpt_path)
    state = torch.load(ckpt_path)
    lm_model.load_state_dict(state["lm_model"])

  params = list(lm_model.parameters())
  opt, sche = optim_utils.get_optimizer(
      FLAGS.lm_opt, FLAGS.lm_lr, FLAGS.lm_decay, FLAGS.lm_step, params)

  global_step = 0
  stats = new_stats()
  for seq, mask, tok_type, label, lm_label in tqdm.tqdm(
          dataloader, total=FLAGS.lm_step):
    opt.zero_grad()

    global_step += 1

    seq = seq.to(DEVICE)
    mask = mask.to(DEVICE)
    tok_type = tok_type.to(DEVICE)
    label = label.to(DEVICE)
    lm_label = lm_label.to(DEVICE)
    is_summary_iter = global_step % (FLAGS.lm_period_summary * 10) == 0

    if is_summary_iter:
      stats["example_seq"] = seq.detach().cpu().numpy()
      stats["example_mask"] = mask.detach().cpu().numpy()
      stats["example_label"] = label.detach().cpu().numpy()
      stats["example_lm_label"] = lm_label.detach().cpu().numpy()

    lm_loss = compute_lm_loss(
        lm_model, seq, mask, tok_type, lm_label, stats, is_summary_iter)
    lm_loss.backward()

    opt.step()
    sche.step()

    if global_step % FLAGS.lm_period_summary == 0:
      write_summary(lm_model, dataset._tokenizer, stats, summary,
                    global_step, is_summary_iter)
      stats = new_stats()

    if global_step % FLAGS.lm_period_save == 0:
      save_model(lm_model, opt, global_step, output_dir)

    if global_step >= FLAGS.lm_step:
      break
