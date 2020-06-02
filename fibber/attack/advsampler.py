import numpy as np
import tqdm

import stanza
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertForMaskedLM, BertTokenizer

from .. import log
from .lm import prepare_lm
from .wordpiece_emb import get_wordpiece_emb

logger = log.setup_custom_logger('adv')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tostring(tokenizer, seq):
  return tokenizer.convert_tokens_to_string(
      tokenizer.convert_ids_to_tokens(seq))


def compute_clf(clf_model, seq, tok_type):
  return clf_model(seq.unsqueeze(0), token_type_ids=tok_type.unsqueeze(0)
                   )[0].argmax(dim=1)[0].detach().cpu().numpy()


def load_model(model_init, checkpoint):
  lm_model = BertForMaskedLM.from_pretrained(model_init)
  state = torch.load(checkpoint)
  lm_model.load_state_dict(state["lm_model"])
  lm_model = lm_model.to(DEVICE)
  return lm_model


def compute_emb_word_prob(word_embs, current, mask,
                          target, eps, smooth, st, ed):
  emb = (word_embs(current[:, st:ed]) * mask[:, st:ed, None])
  emb = emb.sum(dim=1)  # batch * dim
  emb = (emb[:, None, :] + word_embs.weight[None, :, :])  # batch * vocab * dim

  dis = F.cosine_similarity(target[:, None, :], emb, dim=2)  # batch * vocab
  dis = (eps - dis).clamp_(min=0)
  logpdf = -smooth * dis
  return logpdf.detach().to(DEVICE)


def get_emb(word_embs, seq, st, ed):
  return (word_embs(seq[st:ed])).sum(dim=0)


def compute_sim(x, y):
  return F.cosine_similarity(x, y, dim=0)


def sample_text(lm_model, word_embs, keep_words, seq, tok_type, target,
                MASK_ID, FLAGS, pbar, st, ed):
  ret = []
  seq = seq.clone().unsqueeze(0)
  target = target.unsqueeze(0)
  tok_type = tok_type.unsqueeze(0)
  mask_t = torch.ones_like(seq)
  mask_t[:, 0] = 0

  keep_words = (F.one_hot(seq * tok_type, keep_words.size(0))
                * keep_words).sum(dim=1)
  # 1 * vocab_size

  seq_len = ed - st
  block_size = FLAGS.gibbs_block

  for i in range(FLAGS.gibbs_iter):
    if i < FLAGS.gibbs_iter / 2:
      eps = FLAGS.gibbs_eps1
    else:
      eps = FLAGS.gibbs_eps2
    pos = np.arange(st, ed)
    if FLAGS.gibbs_order == "rand":
      np.random.shuffle(pos)
    else:
      assert FLAGS.gibbs_order == "seq"

    max_step = (seq_len - 2) // + 1
    for step in range(max_step):
      pos_t = pos[step * block_size: step * block_size + block_size]

      for j, p in enumerate(pos_t):
        if keep_words[0][seq[0, p]] == 1:
          pos_t[j] = -1
        else:
          if keep_words[0][seq[0, p]] > 0:
            keep_words[0][seq[0, p]] -= 1
          seq[:, p] = MASK_ID
          mask_t[:, p] = 0

      for p in pos_t:
        if p == -1:
          continue
        logits = lm_model(seq, token_type_ids=tok_type)[0][:, p]
        logits2 = compute_emb_word_prob(word_embs, seq, mask_t, target,
                                        eps, FLAGS.gibbs_smooth, st, ed)

        if FLAGS.gibbs_topk > 0:
          topk_v, topk_id = torch.topk(logits + logits2, FLAGS.gibbs_topk)
          dist = torch.distributions.categorical.Categorical(logits=topk_v)
          idx = dist.sample()
          idx = torch.gather(topk_id, dim=1, index=idx.unsqueeze(1)).squeeze(1)
        else:
          dist = torch.distributions.categorical.Categorical(
              logits=logits + logits2)
          idx = dist.sample()

        seq[:, p] = idx
        mask_t[:, p] = 1
        if keep_words[0][idx[0]] > 0:
          keep_words[0][idx[0]] += 1

    ret.append(seq[0].clone().detach())
    pbar.update(1)
  return ret


def attack(lm_model, clf_model, word_embs, keep_words, seq, tok_type, label,
           MASK_ID, FLAGS, pbar, st, ed):
  target_emb = get_emb(word_embs, seq, st, ed)

  sampled_seq = sample_text(
      lm_model, word_embs, keep_words, seq, tok_type, target_emb,
      MASK_ID, FLAGS, pbar, st, ed)

  clf_labels = [compute_clf(clf_model, item, tok_type) for item in sampled_seq]

  best = None
  best_sim = -1
  best_label = None

  for seq_t, label_t in zip(sampled_seq, clf_labels):
    if label_t != label:
      sim_t = compute_sim(target_emb, get_emb(word_embs, seq_t, st, ed))
      if sim_t > best_sim:
        best_sim = sim_t
        best = seq_t
        best_label = label_t

  return best, best_label


class AdvSampler(object):

  def __init__(self, FLAGS, trainset, testset):
    super(AdvSampler, self).__init__()

    logger.info("train language models.")
    prepare_lm(FLAGS, trainset, filter=-1)
    if FLAGS.attack_method == "adv":
      for i in range(len(trainset["label_mapping"])):
        prepare_lm(FLAGS, trainset, filter=i)

    logger.info("train word piece embeddings.")
    self._wpe = get_wordpiece_emb(FLAGS, trainset)

    with open(FLAGS.data_dir + "/stopwords.txt") as f:
      self._stopwords = f.readlines()
      self._stopwords = [w.lower().strip() for w in self._stopwords]

  def attack_clf(self, FLAGS, attackset, clf_model):
    if attackset["cased"]:
      model_init = "bert-base-cased"
    else:
      model_init = "bert-base-uncased"

    num_label = len(attackset["label_mapping"])

    # init tokenizer and models
    tokenizer = BertTokenizer.from_pretrained(model_init)
    PAD_ID = tokenizer.pad_token_id
    CLS_ID = tokenizer.cls_token_id
    MASK_ID = tokenizer.mask_token_id

    if FLAGS.gibbs_keep_entity == "1":
      stanza_ner = stanza.Pipeline(lang='en', processors='tokenize,ner')

    data = [[] for i in range(num_label)]
    for idx, item in enumerate(attackset["data"]):
      data[item["label"]].append(idx)

    count = 0
    correct_ori = 0
    correct_adv = 0

    pbar = tqdm.tqdm(total=len(attackset["data"]) *
                     FLAGS.gibbs_round * FLAGS.gibbs_iter)
    attacker_name = (
        "/{method}-round{round}-iter{iter}-block{block}-eps{eps1}~{eps2}"
        "-smooth{smooth}-{order}-top{topk}".format(
            method=FLAGS.attack_method, round=FLAGS.gibbs_round,
            iter=FLAGS.gibbs_iter, block=FLAGS.gibbs_block,
            eps1=FLAGS.gibbs_eps1, eps2=FLAGS.gibbs_eps2,
            smooth=FLAGS.gibbs_smooth, order=FLAGS.gibbs_order,
            topk=FLAGS.gibbs_topk))

    word_embs = nn.Embedding(tokenizer.vocab_size, 300)
    word_embs.weight.data = torch.tensor(self._wpe.T).float()
    word_embs = word_embs.to(DEVICE)

    result = []

    for label in range(len(attackset["label_mapping"])):
      if FLAGS.attack_method == "all":
        if label == 0:
          lm_model = load_model(
              model_init,
              FLAGS.output_dir + "/%s/model/checkpoint-%04dk.pt" % (
                  "lm_all", FLAGS.lm_step // 1000))
      elif FLAGS.attack_method == "adv":
        lm_model = load_model(
            model_init,
            FLAGS.output_dir
            + "/%s/model/checkpoint-%04dk.pt" % (
                "lm_no_c%d" % label, FLAGS.lm_step // 1000))
      else:
        assert 0

      for idx in data[label]:
        if count > 0:
          pbar.set_postfix(ori_acc="%.1f" % (100 * correct_ori / count),
                           adv_acc="%.1f" % (100 * correct_adv / count),)
        count += 1

        data_record = attackset["data"][idx]
        assert data_record["label"] == label

        seq = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("[CLS] " + data_record["s0"]))
        l0 = len(seq)
        tok_type = [0] * len(seq)

        seq = torch.tensor(seq[:FLAGS.gibbs_max_len]).to(DEVICE)
        tok_type = torch.tensor(tok_type[:FLAGS.gibbs_max_len]).to(DEVICE)
        l0 = min(l0, FLAGS.gibbs_max_len)

        keep_words = np.zeros(tokenizer.vocab_size, dtype="int8")
        if FLAGS.gibbs_keep_entity == "1":
          for sent in stanza_ner(data_record["s0"]).sentences:
            for ent in sent.ents:
              if ent.type not in ["PERSON", "ORG", "LOC"]:
                continue
              for ent_tok in tokenizer.tokenize(ent.text):
                if ent_tok.lower() not in self._stopwords:
                  keep_words[tokenizer.vocab[ent_tok]] = 1
        keep_words = torch.tensor(keep_words).int().to(DEVICE)

        ori_pred = compute_clf(clf_model, seq, tok_type)

        if ori_pred != label:
          result.append({
              "id": idx,
              "label": label,
              "ori": {
                  "s0": data_record["s0"],
                  "pred": ori_pred,
              }
          })
          pbar.update(FLAGS.gibbs_iter * FLAGS.gibbs_round)
          continue
        else:
          correct_ori += 1

        adv_seq = None
        for _ in range(FLAGS.gibbs_round):
          if adv_seq is None:
            adv_seq, adv_label = attack(
                lm_model, clf_model, word_embs, keep_words, seq, tok_type,
                label, MASK_ID, FLAGS, pbar, 1, l0)
          else:
            pbar.update(FLAGS.gibbs_iter)

        if adv_seq != None:
          result.append({
              "id": idx,
              "label": label,
              "ori": {
                  "s0": data_record["s0"],
                  "pred": label,
              },
              "adv": {
                  "s0": tostring(tokenizer, adv_seq[1:]),
                  "pred": int(adv_label)
              }
          })
        else:
          result.append({
              "id": idx,
              "label": label,
              "ori": {
                  "s0": data_record["s0"],
                  "pred": label,
              },
          })
          correct_adv += 1
    result = sorted(result, key=lambda x: x["id"])
    return attacker_name, result

  def attack_nli(self, FLAGS, attackset, clf_model):
    if attackset["cased"]:
      model_init = "bert-base-cased"
    else:
      model_init = "bert-base-uncased"

    num_label = len(attackset["label_mapping"])

    # init tokenizer and models
    tokenizer = BertTokenizer.from_pretrained(model_init)
    PAD_ID = tokenizer.pad_token_id
    CLS_ID = tokenizer.cls_token_id
    MASK_ID = tokenizer.mask_token_id

    if FLAGS.gibbs_keep_entity == "1":
      stanza_ner = stanza.Pipeline(lang='en', processors='tokenize,ner')

    data = [[] for i in range(num_label)]
    for idx, item in enumerate(attackset["data"]):
      data[item["label"]].append(idx)

    count = 0
    correct_ori = 0
    correct_adv = 0

    pbar = tqdm.tqdm(total=len(attackset["data"]) *
                     FLAGS.gibbs_round * FLAGS.gibbs_iter)
    attacker_name = (
        "/{method}-round{round}-iter{iter}-block{block}-eps{eps1}~{eps2}"
        "-smooth{smooth}-{order}-top{topk}".format(
            method=FLAGS.attack_method, round=FLAGS.gibbs_round,
            iter=FLAGS.gibbs_iter, block=FLAGS.gibbs_block,
            eps1=FLAGS.gibbs_eps1, eps2=FLAGS.gibbs_eps2,
            smooth=FLAGS.gibbs_smooth, order=FLAGS.gibbs_order,
            topk=FLAGS.gibbs_topk))

    word_embs = nn.Embedding(tokenizer.vocab_size, 300)
    word_embs.weight.data = torch.tensor(self._wpe.T).float()
    word_embs = word_embs.to(DEVICE)

    result = []

    for label in range(len(attackset["label_mapping"])):
      if FLAGS.attack_method == "all":
        if label == 0:
          lm_model = load_model(
              model_init,
              FLAGS.output_dir + "/%s/model/checkpoint-%04dk.pt" % (
                  "lm_all", FLAGS.lm_step // 1000))
      elif FLAGS.attack_method == "adv":
        lm_model = load_model(
            model_init,
            FLAGS.output_dir
            + "/%s/model/checkpoint-%04dk.pt" % (
                "lm_no_c%d" % label, FLAGS.lm_step // 1000))
      else:
        assert 0

      for idx in data[label]:
        if count > 0:
          pbar.set_postfix(ori_acc="%.1f" % (100 * correct_ori / count),
                           adv_acc="%.1f" % (100 * correct_adv / count),)
        count += 1

        data_record = attackset["data"][idx]
        assert data_record["label"] == label

        s0 = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("[CLS] " + data_record["s0"]))
        s1 = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("[SEP] " + data_record["s1"]))
        l0 = len(s0)
        l1 = len(s1)
        seq = s0 + s1
        tok_type = [0] * l0 + [1] * l1

        seq = torch.tensor(seq[:FLAGS.gibbs_max_len]).to(DEVICE)
        tok_type = torch.tensor(tok_type[:FLAGS.gibbs_max_len]).to(DEVICE)
        l0 = min(l0, FLAGS.gibbs_max_len)
        l1 = min(l1, FLAGS.gibbs_max_len - l0)

        keep_words = np.zeros(tokenizer.vocab_size, dtype="int8")
        if FLAGS.gibbs_keep_entity == "1":
          for sent in stanza_ner(data_record["s1"]).sentences:
            for ent in sent.ents:
              if ent.type not in ["PERSON", "ORG", "LOC"]:
                continue
              for ent_tok in tokenizer.tokenize(ent.text):
                if ent_tok.lower() not in self._stopwords:
                  keep_words[tokenizer.vocab[ent_tok]] = 1
        keep_words = torch.tensor(keep_words).int().to(DEVICE)

        ori_pred = compute_clf(clf_model, seq, tok_type)

        if ori_pred != label:
          result.append({
              "id": idx,
              "label": label,
              "ori": {
                  "s0": data_record["s0"],
                  "s1": data_record["s1"],
                  "pred": ori_pred,
              }
          })
          pbar.update(FLAGS.gibbs_iter * FLAGS.gibbs_round)
          continue
        else:
          correct_ori += 1

        adv_seq = None
        for _ in range(FLAGS.gibbs_round):
          if adv_seq is None:
            adv_seq, adv_label = attack(
                lm_model, clf_model, word_embs, keep_words, seq, tok_type,
                label, MASK_ID, FLAGS, pbar, l0 + 1, l0 + l1)
          else:
            pbar.update(FLAGS.gibbs_iter)

        if adv_seq != None:
          result.append({
              "id": idx,
              "label": label,
              "ori": {
                  "s0": data_record["s0"],
                  "s1": data_record["s1"],
                  "pred": label,
              },
              "adv": {
                  "s0": tostring(tokenizer, adv_seq[1:l0]),
                  "s1": tostring(tokenizer, adv_seq[l0+1:]),
                  "pred": int(adv_label)
              }
          })
        else:
          result.append({
              "id": idx,
              "label": label,
              "ori": {
                  "s0": data_record["s0"],
                  "s1": data_record["s1"],
                  "pred": label,
              },
          })
          correct_adv += 1
    result = sorted(result, key=lambda x: x["id"])
    return attacker_name, result
