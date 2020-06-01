import logging
import os

import numpy as np
import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertTokenizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_glove_model(glove_file, dim=300):
  glove_file_lines = open(glove_file, 'r').readlines()

  emb_table = np.zeros((len(glove_file_lines), dim), dtype='float32')
  id_to_tok = []
  tok_to_id = {}

  logging.info("load glove embeddings for learning word piece embeddings.")
  id = 0
  for line in tqdm.tqdm(glove_file_lines):
    split_line = line.split()
    word = split_line[0]
    emb_table[id] = np.array([float(val) for val in split_line[1:]])
    id_to_tok.append(word)
    tok_to_id[word] = id
    id += 1

  return emb_table, id_to_tok, tok_to_id


class WordPieceDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, model_init, data_dir):
    self._tokenizer = BertTokenizer.from_pretrained(model_init)

    self._glove_emb, self._glove_id2str, self._glove_tok2id = load_glove_model(
        data_dir + "/glove.6B.300d.txt", 300)

    with open(data_dir + "/stopwords.txt") as f:
      stopwords = f.readlines()
    for word in stopwords:
      word = word.lower().strip()
      if word in self._glove_tok2id:
        self._glove_emb[self._glove_tok2id[word], :] = 0

    data = []
    logging.info("processing data for wordpiece embedding training")
    for item in tqdm.tqdm(dataset["data"]):
      text = item["s0"]
      if "s1" in item:
        text += " " + item["s1"]

      text_toks = self._tokenizer.convert_tokens_to_string(
          self._tokenizer.tokenize(text)).split()
      data += [x for x in text_toks if x.lower() in self._glove_tok2id]
    self._data = data

  def __len__(self):
    return len(self._data)

  def __getitem__(self, i):
    w = self._data[i]

    g_emb = self._glove_emb[self._glove_tok2id[w.lower()]]

    comp = np.zeros(self._tokenizer.vocab_size)
    for tok_id in self._tokenizer.convert_tokens_to_ids(
            self._tokenizer.tokenize(w)):
      comp[tok_id] = 1

    return torch.tensor(comp).float(), torch.tensor(g_emb).float()


def get_wordpiece_emb(FLAGS, trainset):
  filename = FLAGS.output_dir + "/wordpiece_emb-%04dk.pt" % (
      FLAGS.wpe_step // 1000)
  if os.path.exists(filename):
    state_dict = torch.load(filename)
    logging.info("load wordpiece embeddings from %s", filename)
    return state_dict["embs"]

  if trainset["cased"]:
    model_init = "bert-base-cased"
  else:
    model_init = "bert-base-uncased"

  dataset = WordPieceDataset(trainset, model_init, FLAGS.data_dir)
  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=FLAGS.wpe_bs, shuffle=True,
      num_workers=5, drop_last=True)

  linear = nn.Linear(dataset._tokenizer.vocab_size, 300, bias=False).to(DEVICE)
  opt = torch.optim.SGD(lr=FLAGS.wpe_lr, momentum=FLAGS.wpe_momentum,
                        params=linear.parameters())
  scheduler = torch.optim.lr_scheduler.StepLR(
      opt, step_size=FLAGS.wpe_peroid_lr_halve, gamma=0.5)

  for w, wid in dataset._tokenizer.vocab.items():
    if w.lower() in dataset._glove_tok2id:
      linear.weight.data[:, wid] = torch.tensor(
          dataset._glove_emb[dataset._glove_tok2id[w.lower()]]).to(DEVICE)

  logging.info("train word piece embeddings")
  pbar = tqdm.tqdm(total=FLAGS.wpe_step)
  losses = []
  global_step = 0
  while True:
    for idx, (comp, target) in enumerate(dataloader):
      comp = comp.to(DEVICE)
      target = target.to(DEVICE)
      pred_emb = linear(comp)
      loss = ((pred_emb - target).abs()).sum(dim=1).mean(dim=0)
      losses.append(loss.detach().cpu().numpy())
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      losses = losses[-20:]

      pbar.set_postfix(loss=np.mean(losses), std=target.std())
      pbar.update(1)
      global_step += 1
      if global_step >= FLAGS.wpe_step:
        break
    if global_step >= FLAGS.wpe_step:
      break

  pbar.close()
  wordpiece_emb = linear.weight.data.cpu().numpy()
  torch.save({"embs": wordpiece_emb}, filename)
  return wordpiece_emb
