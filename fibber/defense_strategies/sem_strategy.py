import copy
import json
import os
from collections import Counter
from functools import partial

import numpy as np
import torch
import tqdm
from nltk import word_tokenize

from fibber.defense_strategies.defense_strategy_base import DefenseStrategyBase
from fibber.metrics.classifier.input_manipulation_classifier import InputManipulationClassifier
from fibber.resources.resource_utils import get_counter_fitted_vector


def load_or_build_sem_wordmap(save_path, trainset, field, device, kk=10, delta=0.5):
    """Load or build the synonym encoding.

     See Natural Language Adversarial Defense through Synonym Encoding
     (https://arxiv.org/abs/1909.06723)

     Args:
         dataset_name (str): name of the dataset.
         trainset (dict): the training set. (used to compute word frequency.)
         kk (int): maximum synonym considered for each word.
         delta (float): threshold for synonym.
     Returns:
         (dict): a map from word to encoding.
     """
    wordmap_filename = os.path.join(save_path, "sem_wordmap_k_%d_delta_%f.json" % (kk, delta))

    if os.path.exists(wordmap_filename):
        with open(wordmap_filename) as f:
            word_map = json.load(f)
        return word_map

    emb = get_counter_fitted_vector()
    freq = Counter(emb["id2tok"])

    for item in trainset["data"]:
        freq.update(word_tokenize(item[field].lower()))
    word_by_freq = sorted(list(freq.items()), key=lambda x: -x[1])
    word_by_freq = [item for item in word_by_freq if item[0] in emb["tok2id"]]
    word_map = dict([(item[0], None) for item in word_by_freq])

    emb_table_tensor = torch.tensor(emb["emb_table"]).float().to(device)

    def find_syns(word):
        word_id = emb["tok2id"][word]
        dis_t = torch.sum((emb_table_tensor - emb_table_tensor[word_id]) ** 2, dim=1)
        dis_t = dis_t.detach().cpu().numpy()
        syn_ids = np.argsort(dis_t)[:kk]
        tmp = kk
        while tmp >= 1 and dis_t[syn_ids[tmp - 1]] > delta:
            tmp -= 1
        return [emb["id2tok"][syn_id] for syn_id in syn_ids[:tmp]]

    for word, freq in word_by_freq[:16]:
        word_map[word] = word

    for word, freq in tqdm.tqdm(word_by_freq):
        if word_map[word] is None:
            syn_list = find_syns(word)

            for syn in syn_list:
                if word_map[syn] is not None:
                    word_map[word] = syn
                    break
            if word_map[word] is None:
                word_map[word] = word

            for syn in syn_list:
                if word_map[syn] is None:
                    word_map[syn] = word_map[word]

    with open(wordmap_filename, "w") as f:
        json.dump(word_map, f, indent=2)
    return word_map


def sem_substitute(tok, word_map):
    if tok.lower() not in word_map:
        return None

    tmp = tok.lower()
    while tmp != word_map[tmp]:
        tmp = word_map[tmp.lower()]
    if 'A' <= tok[0] <= 'Z':
        return tmp.capitalize()
    else:
        return tmp


def sem_fix_sentences(sentences, data_record_list, word_map, reformat=False):
    ret = []
    for sentence in sentences:
        toks = word_tokenize(sentence)
        toks = [sem_substitute(tok, word_map) for tok in toks]
        toks = [tok for tok in toks if tok is not None]
        if reformat:
            ret.append([" ".join(toks)])
        else:
            ret.append(" ".join(toks))
    return ret


def sem_transform_dataset(dataset, word_map, field, deepcopy=True):
    if deepcopy:
        dataset = copy.deepcopy(dataset)
    for item in dataset["data"]:
        item[field] = sem_fix_sentences([item[field]], None, word_map)[0]
    return dataset


class SEMStrategy(DefenseStrategyBase):
    """Base class for Tuning strategy"""

    __abbr__ = "sem"
    __hyperparameters__ = [
        ("steps", int, 5000, "number of rewrites."),
        ("bs", int, 32, "classifier training batch size.")]

    def fit(self, trainset):
        self._sem_wordmap = load_or_build_sem_wordmap(
            self._defense_save_path, trainset, field=self._field, device=self._device)
        trainset_transformed = sem_transform_dataset(trainset, self._sem_wordmap, self._field)

        self._classifier.robust_tune_init(optimizer="adamw", lr=1e-5, weight_decay=0.001,
                                          steps=self._strategy_config["steps"])

        for _ in tqdm.tqdm(list(range(self._strategy_config["steps"]))):
            batch = np.random.choice(trainset_transformed["data"], self._strategy_config["bs"])
            self._classifier.robust_tune_step(batch)
        self._classifier.save_robust_tuned_model(self._defense_save_path)

    def load(self, trainset):
        self._sem_wordmap = load_or_build_sem_wordmap(
            self._defense_save_path, trainset, field=self._field, device=self._device)
        self._classifier.load_robust_tuned_model(self._defense_save_path)
        return InputManipulationClassifier(
            self._classifier,
            partial(sem_fix_sentences, word_map=self._sem_wordmap, reformat=True),
            str(self._classifier), field=self._classifier._field, bs=self._classifier._bs)
