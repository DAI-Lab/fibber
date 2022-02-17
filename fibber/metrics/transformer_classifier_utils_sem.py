import copy
import json
import os
from collections import Counter

import numpy as np
import torch
import tqdm
from nltk.tokenize import word_tokenize

from fibber import get_root_dir
from fibber.resources import get_counter_fitted_vector


def load_or_build_sem_wordmap(dataset_name, trainset, device, kk=10, delta=0.5):
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
    os.makedirs(os.path.join(get_root_dir(), "sem_wordmap"), exist_ok=True)
    wordmap_filename = os.path.join(get_root_dir(), "sem_wordmap",
                                    "%s_k_%d_delta_%f.json" % (dataset_name, kk, delta))

    if os.path.exists(wordmap_filename):
        with open(wordmap_filename) as f:
            word_map = json.load(f)
        return word_map

    emb = get_counter_fitted_vector()
    freq = Counter(emb["id2tok"])

    for item in trainset["data"]:
        freq.update(word_tokenize(item[trainset["paraphrase_field"]].lower()))
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


def sem_fix_sentences(sentences, word_map):
    ret = []
    for sentence in sentences:
        toks = word_tokenize(sentence)
        toks = [sem_substitute(tok, word_map) for tok in toks]
        toks = [tok for tok in toks if tok is not None]
        ret.append(" ".join(toks))
    return ret


def sem_transform_dataset(dataset, word_map, deepcopy=True):
    if deepcopy:
        dataset = copy.deepcopy(dataset)
    field_name = dataset["paraphrase_field"]
    for item in dataset["data"]:
        item[field_name] = sem_fix_sentences([item[field_name]], word_map)[0]
    return dataset
