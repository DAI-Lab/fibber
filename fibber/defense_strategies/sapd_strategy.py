import copy

import numpy as np
import torch
import tqdm
from nltk.tokenize import word_tokenize
from transformers import (
    BertForSequenceClassification, DistilBertForSequenceClassification,
    RobertaForSequenceClassification)

from fibber import log
from fibber.defense_strategies.defense_strategy_base import DefenseStrategyBase

logger = log.setup_custom_logger(__name__)

INFTY = 1e8


class EmbeddingGradHook:
    def __init__(self):
        self.output = None
        self._enable = True

    def disable(self):
        self._enable = False
        self.output = None

    def enable(self):
        self._enable = True

    def __call__(self, module, input_, output_):
        if self._enable:
            self.output = output_
            output_.retain_grad()
        else:
            self.output = None


def random_perturb(data_record_list, tokenizer, vocabulary, field):
    data_record_list = copy.deepcopy(data_record_list)
    for i in range(len(data_record_list)):
        text = data_record_list[i][field]
        toks = tokenizer.tokenize(text)
        pos = np.random.randint(0, len(toks))
        toks[pos] = vocabulary[np.random.randint(len(vocabulary))][0]
        data_record_list[i][field] = tokenizer.convert_tokens_to_string(toks)
    return data_record_list


def random_long_word_perturb(data_record_list, tokenizer, adv_long_word, field):
    data_record_list = copy.deepcopy(data_record_list)
    for i in range(len(data_record_list)):
        text = data_record_list[i][field]
        label = data_record_list[i]["label"]
        sent_toks = tokenizer.tokenize(text)
        wid = np.random.choice(len(adv_long_word[label]))
        long_word_toks = adv_long_word[label][wid]
        pos = np.random.randint(len(sent_toks))
        sent_toks = sent_toks[:pos] + long_word_toks + sent_toks[pos + 1:]
        data_record_list[i][field] = tokenizer.convert_tokens_to_string(sent_toks)

    return data_record_list


def grad_perturb(data_record_list, tokenizer, vocabulary, model, lm_model, device, field, topk=1):
    data_record_list = copy.deepcopy(data_record_list)
    n = len(data_record_list)
    if field != "text0":
        raise RuntimeError
    if "text1" in data_record_list[0]:
        raise RuntimeError

    batch_input = tokenizer(
        [item["text0"] for item in data_record_list],
        return_tensors="pt",
        padding=True)
    seq = batch_input["input_ids"]
    mask = batch_input["attention_mask"]
    label = torch.tensor([item["label"] for item in data_record_list])

    embedding_layer = lm_model.embeddings.word_embeddings
    hook = EmbeddingGradHook()
    embedding_layer.register_forward_hook(hook)

    output = model(input_ids=seq.to(device), attention_mask=mask.to(device),
                   labels=label.to(device), return_dict=True)

    (-output["loss"]).sum().backward()
    grad = hook.output.grad.clone()    # batch * L * d
    emb_output = hook.output.clone()   # batch * L * d

    with torch.no_grad():
        hook.disable()
        embs_all = embedding_layer(torch.tensor([item[1] for item in vocabulary]).to(device))
        weight = (torch.einsum("vd,bld->blv", embs_all, grad)
                  - torch.einsum("bld,bld->bl", grad, emb_output).unsqueeze(2))
        weight += (INFTY * (1 - mask.to(device))).unsqueeze(2)
        weight[:, 0] = INFTY
        for i in range(n):
            weight[i, torch.sum(mask[i]) - 1] = INFTY
        args = torch.argsort(weight.reshape(n, -1), dim=1).detach().cpu().numpy()

    for i in range(n):
        if topk == 1:
            rand_pos = 0
        else:
            rand_pos = np.random.randint(topk)
        u = args[i][rand_pos] // len(vocabulary)
        v = args[i][rand_pos] % len(vocabulary)

        tmp = seq[i]
        tmp[u] = vocabulary[v][1]
        data_record_list[i][field] = tokenizer.decode(tmp[1:torch.sum(mask[i]) - 1])

    return data_record_list


def get_adv_long_words(trainset, tokenizer):
    word_counter = {}
    n_label = len(trainset["label_mapping"])

    for item in tqdm.tqdm(trainset["data"]):
        toks = word_tokenize(item["text0"])
        for tok in toks:
            if len(tok) > 20:
                continue
            try:
                word_counter[tok][item["label"]] += 1
            except KeyError:
                word_counter[tok] = [0] * n_label

    ret = []
    for label in range(n_label):
        tok_score = []
        for tok, cnt in word_counter.items():
            n_current = np.log(cnt[label] + 1)
            n_other = np.log(np.max(cnt) + 1)
            score = n_other - n_current
            if score > 1:
                tmp = tokenizer.tokenize(tok)
                if len(tmp) > 5:
                    continue
                tok_score.append((tmp, score))
        tok_score = sorted(tok_score, key=lambda x: -x[1])[:2000]
        ret.append([k for k, v in tok_score])
    return ret


def special_word(x):
    if len(x) == 0:
        return True
    if x[0] == "[" and x[-1] == "]":
        return True
    if x[0] == "<" and x[-1] == ">":
        return True
    return False


class SAPDStrategy(DefenseStrategyBase):
    """Base class for Tuning strategy"""

    __abbr__ = "sapd"
    __hyperparameters__ = [
        ("steps", int, 5000, "number of rewrites."),
        ("bs", int, 32, "classifier training batch size.")]

    def fit(self, trainset):
        steps = self._strategy_config["steps"]
        bs = self._strategy_config["bs"]

        if isinstance(self._classifier._model, DistilBertForSequenceClassification):
            lm_model = self._classifier._model.distilbert
        elif isinstance(self._classifier._model, BertForSequenceClassification):
            lm_model = self._classifier._model.bert
        elif isinstance(self._classifier._model, RobertaForSequenceClassification):
            lm_model = self._classifier._model.roberta
        else:
            raise RuntimeError("unknown victim.")

        tokenizer = self._classifier._tokenizer
        vocabulary = list(tokenizer.vocab.items())
        vocabulary = sorted([item for item in vocabulary if not special_word(item[0])],
                            key=lambda x: x[1])

        adv_long_words = get_adv_long_words(trainset, tokenizer)

        self._classifier.robust_tune_init(optimizer="adamw", lr=1e-5, weight_decay=0.001,
                                          steps=steps)
        for i in tqdm.trange(steps):
            batch = list(np.random.choice(trainset["data"], bs // 2))
            if i % 3 == 0:
                batch = random_perturb(batch, tokenizer, vocabulary, self._field)
            elif i % 3 == 1:
                batch = random_long_word_perturb(batch, tokenizer, adv_long_words, self._field)
            batch_perturb = grad_perturb(
                batch, tokenizer, vocabulary, model=self._classifier._model,
                lm_model=lm_model, device=self._classifier._device, field=self._field)

            batch_all = batch + batch_perturb
            self._classifier.robust_tune_step(batch_all)

        self._classifier.save_robust_tuned_model(self._defense_save_path)

    def load(self, trainset):
        self._classifier.load_robust_tuned_model(self._defense_save_path)
        return self._classifier
