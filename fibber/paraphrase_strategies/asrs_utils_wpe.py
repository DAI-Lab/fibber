import os

import numpy as np
import torch
import tqdm
from nltk import word_tokenize
from torch import nn

from fibber import get_root_dir, log
from fibber.resources import get_counter_fitted_vector

logger = log.setup_custom_logger(__name__)


class WordPieceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self._tokenizer = tokenizer

        self._glove = get_counter_fitted_vector()

        data = []
        logger.info("processing data for wordpiece embedding training")
        for item in tqdm.tqdm(dataset["data"]):
            text = item["text0"]
            if "text1" in item:
                text += " " + item["text1"]

            text_toks = word_tokenize(text)
            data += [x for x in text_toks if x.lower() in self._glove["tok2id"]]
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        w = self._data[i]

        g_emb = self._glove["emb_table"][self._glove["tok2id"][w.lower()]]

        comp = np.zeros(self._tokenizer.vocab_size)
        for tok_id in self._tokenizer.convert_tokens_to_ids(
                self._tokenizer.tokenize(w)):
            comp[tok_id] = 1

        return torch.tensor(comp).float(), torch.tensor(g_emb).float()


def get_wordpiece_emb(dataset_name, trainset, tokenizer, device,
                      steps=5000, bs=1000, lr=1, lr_halve_steps=1000):
    """Transfer GloVe embeddings to BERT vocabulary.

    The transfered embeddings will be stored at ``~.fibber/wordpiece_emb_conterfited/
    wordpiece_emb_<dataset>_<steps>.pt``.

    Args:
        dataset_name (str): dataset name.
        trainset (dict): the dataset dist.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer that specifies wordpieces.
        device (torch.Device): a device to train the model.
        steps (int): transfering steps.
        bs (int): transfering batch size.
        lr (str): transfering learning rate.
        lr_halve_steps (int): steps to halve the learning rate.
    Returns:
        (np.array): a array of size (300, N) where N is the vocabulary size for a bert-base model.
    """
    filename = os.path.join(get_root_dir(), "wordpiece_emb_conterfited")
    os.makedirs(filename, exist_ok=True)
    filename = os.path.join(filename, "wordpiece_emb-%s-%04d.pt" % (dataset_name, steps))
    if os.path.exists(filename):
        state_dict = torch.load(filename)
        logger.info("load wordpiece embeddings from %s", filename)
        return state_dict["embs"]

    dataset = WordPieceDataset(trainset, tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=True,
        num_workers=0, drop_last=True)

    linear = nn.Linear(tokenizer.vocab_size, 300, bias=False).to(device)
    opt = torch.optim.SGD(lr=lr, momentum=0.5,
                          params=linear.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=lr_halve_steps, gamma=0.5)

    linear.weight.data.zero_()
    cc = 0
    for w, wid in tokenizer.vocab.items():
        if w.lower() in dataset._glove["tok2id"]:
            cc += 1
            linear.weight.data[:, wid] = torch.tensor(
                dataset._glove["emb_table"][dataset._glove["tok2id"][w.lower()]]).to(device)
    logger.info("Overlap bert and counter fitted vectors: %d", cc)

    logger.info("train word piece embeddings")
    pbar = tqdm.tqdm(total=steps)
    losses = []
    global_step = 0
    while True:
        for idx, (comp, target) in enumerate(dataloader):
            comp = comp.to(device)
            target = target.to(device)
            pred_emb = linear(comp)
            loss = ((pred_emb - target).abs()).sum(dim=1).mean(dim=0)
            losses.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            losses = losses[-20:]

            pbar.set_postfix(loss=np.mean(losses), std=target.std().detach().cpu().numpy())
            pbar.update(1)
            global_step += 1
            if global_step >= steps:
                break
        if global_step >= steps:
            break

    pbar.close()
    wordpiece_emb = linear.weight.data.cpu().numpy()
    torch.save({"embs": wordpiece_emb}, filename)
    return wordpiece_emb
