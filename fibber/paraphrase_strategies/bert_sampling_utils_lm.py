import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForMaskedLM, BertTokenizerFast

from fibber import log, resources
from fibber.datasets import DatasetForBert
from fibber.metrics.bert_classifier import get_optimizer

import copy

logger = log.setup_custom_logger(__name__)


class NonAutoregressiveBertLM(BertForMaskedLM):
    def __init__(self, config, sentence_embed_size):
        super().__init__(config)
        self.sentence_emb_transform = nn.Sequential(
            nn.Linear(sentence_embed_size + self.config.hidden_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )

    def _compute_emb(self, sentence_embeds, input_ids):
        # print(sentence_embeds.shape)
        bert_input_embs = self.bert.embeddings.word_embeddings(input_ids)
        transformed_sentence_embs = self.sentence_emb_transform(
            torch.cat([
                bert_input_embs,
                torch.tensor(sentence_embeds).to(self.device)[:, None, :].expand(
                    (-1, bert_input_embs.size(1), -1)),
            ], dim=2))

        return bert_input_embs + transformed_sentence_embs

    def forward(self,
                sentence_embeds=None,
                input_ids=None,
                inputs_embeds=None,
                apply_cls=True,
                **kwargs):
        assert inputs_embeds is None

        inputs_embeds = self._compute_emb(sentence_embeds, input_ids)

        # print("inputs_embeds", inputs_embeds.size())
        # print("attention_mask", kwargs["attention_mask"].size())
        # print("token_type_ids", kwargs["token_type_ids"].size())
        if not apply_cls:
            return self.bert(
                input_ids=None, inputs_embeds=inputs_embeds, **kwargs)
        else:
            return super(NonAutoregressiveBertLM, self).forward(
                input_ids=None, inputs_embeds=inputs_embeds, **kwargs)


class NARRLBertLM(BertForMaskedLM):
    def __init__(self, config, sentence_embed_size):
        super().__init__(config)
        self.sentence_emb_transform = nn.Sequential(
            nn.Linear(sentence_embed_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )

    def _compute_emb(self, sentence_embeds, input_ids):
        bert_input_embs = self.bert.embeddings.word_embeddings(input_ids)
        transformed_sentence_embs = self.sentence_emb_transform(
            torch.tensor(sentence_embeds).to(self.device))
        bert_input_embs[:, 1] = transformed_sentence_embs

        return bert_input_embs

    def forward(self,
                sentence_embeds=None,
                input_ids=None,
                inputs_embeds=None,
                apply_cls=True,
                **kwargs):
        assert inputs_embeds is None

        inputs_embeds = self._compute_emb(sentence_embeds, input_ids)

        if not apply_cls:
            return self.bert(
                input_ids=None, inputs_embeds=inputs_embeds, **kwargs)
        else:
            return super(NARRLBertLM, self).forward(
                input_ids=None, inputs_embeds=inputs_embeds, **kwargs)


def write_summary(stats, summary, global_step):
    """Save langauge model training summary."""
    summary.add_scalar("attack/loss_lm", np.mean(stats["lm_loss"]), global_step)
    summary.add_scalar("attack/error_lm", 1 - stats["lm_correct"] / stats["lm_total"], global_step)


def new_stats():
    """Create a new stats dict."""
    return {
        "lm_total": 0,
        "lm_correct": 0,
        "lm_loss": [],
    }


def compute_lm_loss(lm_model, seq, mask, tok_type, lm_label, stats):
    """Compute masked language model training loss.

    Args:
        lm_model (transformers.BertForMaskedLM): a BERT language model.
        seq (torch.Tensor): an int tensor of size (batch_size, length) representing the word
            pieces.
        mask (torch.Tensor): an int tensor of size (batch_size, length) representing the attention
            mask.
        tok_type (torch.Tensor): an int tensor of size (batch_size, length) representing the
            token type id.
        lm_label (torch.Tensor): an int tensor of size (batch_size, length) representing the label
            for each position. Use -100 if the loss is not computed for that position.
        stats (dist): a dictionary storing training stats.

    Returns:
        (torch.Scalar) a scalar loss value.
    """
    lm_hid = lm_model.bert(seq, mask, tok_type)[0]
    lm_hid = torch.masked_select(lm_hid, lm_label.gt(0).unsqueeze(2)).view(-1, lm_hid.size(2))
    logits = lm_model.cls(lm_hid)

    with torch.no_grad():
        lm_label_squeeze = torch.masked_select(lm_label, lm_label.gt(0))

    lm_loss = F.cross_entropy(logits, lm_label_squeeze)

    # lm_label is -100 for unmasked token or padding toks.
    stats["lm_total"] += (lm_label_squeeze > 0).int().sum().detach().cpu().numpy()
    stats["lm_correct"] += (logits.argmax(dim=1).eq(lm_label_squeeze)
                            .float().sum().detach().cpu().numpy())
    stats["lm_loss"].append(lm_loss.detach().cpu().numpy())

    return lm_loss


def fine_tune_lm(output_dir, trainset, filter, device, lm_steps=5000, lm_bs=32,
                 lm_opt="adamw", lm_lr=0.0001, lm_decay=0.01,
                 lm_period_summary=100, lm_period_save=5000):
    """Returns a finetuned BERT language model on a given dataset.

    The language model will be stored at ``<output_dir>/lm_all`` if filter is -1, or
    ``<output_dir>/lm_filter_?`` if filter is not -1.

    If filter is not -1. The pretrained langauge model will first be pretrained on the while
    dataset, then it will be finetuned on the data excluding the filter category.

    Args:
        output_dir (str): a directory to store pretrained language model.
        trainset (DatasetForBert): the training set for finetune the language model.
        filter (int): a category to exclude from finetuning.
        device (torch.Device): a device to train the model.
        lm_steps (int): finetuning steps.
        lm_bs (int): finetuning batch size.
        lm_opt (str): optimzer name. choose from ["sgd", "adam", "adamW"].
        lm_lr (float): learning rate.
        lm_decay (float): weight decay for the optimizer.
        lm_period_summary (int): number of steps to write training summary.
        lm_period_save (int): number of steps to save the finetuned model.
    Returns:
        (BertForMaskedLM): a finetuned language model.
    """
    if filter == -1:
        output_dir_t = os.path.join(output_dir, "lm_all")
    else:
        output_dir_t = os.path.join(output_dir, "lm_filter_%d" % filter)

    summary = SummaryWriter(output_dir_t + "/summary")

    if trainset["cased"]:
        model_init = "bert-base-cased"
    else:
        model_init = "bert-base-uncased"

    ckpt_path_pattern = output_dir_t + "/checkpoint-%04dk"
    ckpt_path = ckpt_path_pattern % (lm_steps // 1000)

    if os.path.exists(ckpt_path):
        logger.info("Language model <%s> exists.", ckpt_path)
        return BertForMaskedLM.from_pretrained(ckpt_path).eval()

    if filter == -1:
        lm_model = BertForMaskedLM.from_pretrained(resources.get_transformers(model_init))
        lm_model.train()
    else:
        lm_model = get_lm(output_dir, trainset, -1, device, lm_steps)
        lm_model.train()
    lm_model.to(device)

    dataset = DatasetForBert(trainset, model_init, lm_bs, exclude=filter, masked_lm=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=2)

    params = list(lm_model.parameters())
    opt, sche = get_optimizer(lm_opt, lm_lr, lm_decay, lm_steps, params)

    global_step = 0
    stats = new_stats()
    for seq, mask, tok_type, label, lm_label in tqdm.tqdm(
            dataloader, total=lm_steps):
        opt.zero_grad()

        global_step += 1

        seq = seq.to(device)
        mask = mask.to(device)
        tok_type = tok_type.to(device)
        label = label.to(device)
        lm_label = lm_label.to(device)

        lm_loss = compute_lm_loss(
            lm_model, seq, mask, tok_type, lm_label, stats)
        lm_loss.backward()

        opt.step()
        sche.step()

        if global_step % lm_period_summary == 0:
            write_summary(stats, summary, global_step)
            stats = new_stats()

        if global_step % lm_period_save == 0 or global_step == lm_steps:
            lm_model.to(torch.device("cpu")).eval()
            lm_model.save_pretrained(ckpt_path_pattern % (global_step // 1000))
            lm_model.to(device)

        if global_step >= lm_steps:
            break

    lm_model.eval()
    lm_model.to(torch.device("cpu"))
    return lm_model


def compute_non_autoregressive_lm_loss(
        lm_model, sentence_embeds, seq, mask, tok_type, lm_label, stats):
    """Compute masked language model training loss.

    Args:
        lm_model (transformers.BertForMaskedLM): a BERT language model.
        sentence_embeds (np.array): an numpy array of sentence embeddings.
        seq (torch.Tensor): an int tensor of size (batch_size, length) representing the word
            pieces.
        mask (torch.Tensor): an int tensor of size (batch_size, length) representing the attention
            mask.
        tok_type (torch.Tensor): an int tensor of size (batch_size, length) representing the
            token type id.
        lm_label (torch.Tensor): an int tensor of size (batch_size, length) representing the label
            for each position. Use -100 if the loss is not computed for that position.
        stats (dist): a dictionary storing training stats.

    Returns:
        (torch.Scalar) a scalar loss value.
    """

    lm_hid = lm_model(sentence_embeds, input_ids=seq, attention_mask=mask,
                      token_type_ids=tok_type, apply_cls=False)[0]
    lm_hid = torch.masked_select(lm_hid, lm_label.gt(0).unsqueeze(2)).view(-1, lm_hid.size(2))
    logits = lm_model.cls(lm_hid)

    with torch.no_grad():
        lm_label_squeeze = torch.masked_select(lm_label, lm_label.gt(0))

    lm_loss = F.cross_entropy(logits, lm_label_squeeze)

    # lm_label is -100 for unmasked token or padding toks.
    stats["lm_total"] += (lm_label_squeeze > 0).int().sum().detach().cpu().numpy()
    stats["lm_correct"] += (logits.argmax(dim=1).eq(lm_label_squeeze)
                            .float().sum().detach().cpu().numpy())
    stats["lm_loss"].append(lm_loss.detach().cpu().numpy())

    return lm_loss


def non_autoregressive_fine_tune_lm(
        output_dir, trainset, filter, device, use_metric, model_class, num_keywords=0,
        split_sentences=False, lm_steps=5000, lm_bs=32, lm_opt="adamw", lm_lr=0.0001, lm_decay=0.01,
        lm_period_summary=100, lm_period_save=5000, lm_pretune_steps=20000, **kwargs):
    """Returns a finetuned BERT language model on a given dataset.

    The language model will be stored at ``<output_dir>/lm_all`` if filter is -1, or
    ``<output_dir>/lm_filter_?`` if filter is not -1.

    If filter is not -1. The pretrained langauge model will first be pretrained on the while
    dataset, then it will be finetuned on the data excluding the filter category.

    Args:
        output_dir (str): a directory to store pretrained language model.
        trainset (DatasetForBert): the training set for finetune the language model.
        filter (int): a category to exclude from finetuning.
        device (torch.Device): a device to train the model.
        use_metric (USESemanticSimilarityMetric): a sentence encoder metric
        model_class (class): choose from ``[NonAutoregressiveBertLM, NARRLBertLM]``
        split_sentences (bool): whether to train a sentence level language model.
        lm_steps (int): finetuning steps.
        lm_bs (int): finetuning batch size.
        lm_opt (str): optimzer name. choose from ["sgd", "adam", "adamW"].
        lm_lr (float): learning rate.
        lm_decay (float): weight decay for the optimizer.
        lm_period_summary (int): number of steps to write training summary.
        lm_period_save (int): number of steps to save the finetuned model.
    Returns:
        (BertForMaskedLM): a finetuned language model.
    """

    kw_and_padding = (model_class == NARRLBertLM)

    if filter == -1:
        output_dir_t = os.path.join(output_dir, "%s_all" % model_class.__name__)
    else:
        output_dir_t = os.path.join(output_dir, "%s_filter_%d" % (model_class.__name__, filter))

    summary = SummaryWriter(output_dir_t + "/summary")

    if trainset["cased"]:
        model_init = "bert-base-cased"
    else:
        model_init = "bert-base-uncased"

    ckpt_path_pattern = output_dir_t + "/checkpoint-%04dk"
    ckpt_path = ckpt_path_pattern % (lm_steps // 1000)

    if os.path.exists(ckpt_path):
        logger.info("Language model <%s> exists.", ckpt_path)
        return model_class.from_pretrained(ckpt_path, sentence_embed_size=512).eval()

    if filter == -1:
        lm_model = model_class.from_pretrained(resources.get_transformers(model_init),
                                               sentence_embed_size=512)
        lm_model.train()
    else:
        lm_model = get_lm(output_dir, trainset, -1, device, lm_steps)
        lm_model.train()
    lm_model.to(device)

    dataset = DatasetForBert(trainset, model_init, lm_bs, exclude=filter, masked_lm=False,
                             dynamic_masked_lm=True, include_raw_text=True,
                             kw_and_padding=kw_and_padding,
                             split_sentences=split_sentences,
                             num_keywords=num_keywords)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=2)

    params = list(lm_model.parameters())
    opt, sche = get_optimizer(lm_opt, lm_lr, decay=lm_decay,
                              train_step=lm_steps - lm_pretune_steps, params=params)
    pretune_opt = torch.optim.SGD(params=lm_model.sentence_emb_transform.parameters(),
                                  lr=0.01, momentum=0.5)

    global_step = 0
    stats = new_stats()
    for seq, mask, tok_type, label, lm_label, raw_text in tqdm.tqdm(
            dataloader, total=lm_steps):

        if global_step < lm_pretune_steps:
            pretune_opt.zero_grad()
        else:
            opt.zero_grad()

        global_step += 1

        if kw_and_padding:
            raw_text = [x.split("[SEP]")[1] for x in raw_text]
        else:
            raw_text = [x.replace("[CLS]", "").replace("[SEP]", "") for x in raw_text]
        sentence_embs = use_metric.model(raw_text).numpy()

        seq = seq.to(device)
        mask = mask.to(device)
        tok_type = tok_type.to(device)
        label = label.to(device)
        lm_label = lm_label.to(device)

        lm_loss = compute_non_autoregressive_lm_loss(
            lm_model=lm_model, sentence_embeds=sentence_embs, seq=seq, mask=mask,
            tok_type=tok_type, lm_label=lm_label, stats=stats)
        lm_loss.backward()

        if global_step < lm_pretune_steps:
            pretune_opt.step()
        else:
            opt.step()
            sche.step()

        if global_step % lm_period_summary == 0:
            write_summary(stats, summary, global_step)
            stats = new_stats()

        if global_step % lm_period_save == 0 or global_step == lm_steps:
            lm_model.to(torch.device("cpu")).eval()
            lm_model.save_pretrained(ckpt_path_pattern % (global_step // 1000))
            lm_model.to(device)

        if global_step >= lm_steps:
            break

    lm_model.eval()
    lm_model.to(torch.device("cpu"))
    return lm_model


class Env(object):
    def __init__(self, seq, mask, tok_type, raw_text, use_metric, tokenizer,
                 max_step, max_mask_rate, lm_model):
        raw_text = [x.replace("[CLS]", "").replace("[SEP]", "") for x in raw_text]
        self._sentence_embs = use_metric.model(raw_text).numpy()
        self._use_metrc = use_metric
        self._tokenizer = tokenizer
        self._modifiable_pos = mask * (tok_type == 1).long() * (seq != tokenizer.sep_token_id).long()
        self._step = 0

        self._seq = seq
        self._mask = mask
        self._tok_type = tok_type
        self._raw_text = raw_text
        self._text_len = torch.max(tok_type.sum(dim=1)).detach().cpu().numpy()
        self._max_mask_rate = max_mask_rate
        self._max_step = max_step
        self._logits = (torch.rand_like(seq.float()) * self._modifiable_pos
                                 + 1e8 * (1 - self._modifiable_pos))

        # use fixed language model for masking
        # self._lm_model_pretrained = copy.deepcopy(lm_model)

        self._masked_input = None

    def get_state(self):
        num_pos = int(max(self._max_mask_rate * self._text_len
                          * (self._max_step - self._step) / self._max_step, 1))

        tensor_len = self._seq.size(1)
        self._pos = F.one_hot(
            torch.topk(self._logits, k=num_pos, dim=1, largest=False)[1],
            num_classes=tensor_len).sum(dim=1) * self._modifiable_pos

        self._masked_input = self._seq * (1 - self._pos) + self._pos * self._tokenizer.mask_token_id
        return self._masked_input, self._pos

    def step(self, seq_candidate, logits_candidate):
        self._step += 1

        with torch.no_grad():
            # logits = self._lm_model_pretrained(
            #     sentence_embeds=self._sentence_embs,
            #     input_ids=self._seq,
            #     token_type_ids=self._tok_type,
            #     attention_mask=self._mask)[0]
            # logits_candidate_local = torch.gather(
            #     F.log_softmax(logits, dim=2), dim=2, index=seq_candidate[:, :, None]).squeeze(2)

            self._seq = self._pos * seq_candidate + (1 - self._pos) * self._seq
            self._logits = self._pos * logits_candidate + (1 - self._pos) * self._logits

            currect_text = []
            for i in range(self._seq.size(0)):
                t = []
                for wid, modifiable in zip(self._seq[i, :].detach().cpu().numpy(),
                                           self._modifiable_pos[i, :].detach().cpu().numpy()):
                    if modifiable:
                        t.append(wid)
                currect_text.append(self._tokenizer.decode(t))
            print("current text", currect_text)

        if self._step == self._max_step:
            current_emb = self._use_metrc.model(currect_text)
            cos_sim = np.sum((self._sentence_embs / np.linalg.norm(self._sentence_embs, axis=1)[:, None])
                             * (current_emb / np.linalg.norm(current_emb, axis=1)[:, None]), axis=1)

            self._score = cos_sim
            return self._score
        return np.zeros(self._seq.size(0))

    def get_sentence_emb(self):
        return self._sentence_embs


def rl_fine_tune_lm(output_dir, trainset, filter, device, lm_model, use_metric, num_keywords=0,
                    split_sentences=False, num_decode_iter=10, max_mask_rate=1, discount=0.9,
                    rl_steps=5000, rl_bs=32, lm_opt="adamw", lm_lr=0.00001,
                    lm_decay=0, lm_period_summary=100, lm_period_save=5000, **kwargs):
    if filter == -1:
        output_dir_t = os.path.join(output_dir, "%s_rl_all" % lm_model.__class__.__name__)
    else:
        output_dir_t = os.path.join(
            output_dir, "%s_rl_filter_%d" % (lm_model.__class__.__name__, filter))

    summary = SummaryWriter(output_dir_t + "/summary")

    if trainset["cased"]:
        model_init = "bert-base-cased"
    else:
        model_init = "bert-base-uncased"

    ckpt_path_pattern = output_dir_t + "/checkpoint-%04dk"
    ckpt_path = ckpt_path_pattern % (rl_steps // 1000)

    if os.path.exists(ckpt_path):
        logger.info("Language model <%s> exists.", ckpt_path)
        return NARRLBertLM.from_pretrained(ckpt_path, sentence_embed_size=512).eval()

    lm_model.train()
    lm_model.to(device)

    dataset = DatasetForBert(trainset, model_init, rl_bs, exclude=filter, masked_lm=False,
                             dynamic_masked_lm=True, include_raw_text=True,
                             kw_and_padding=True,
                             split_sentences=split_sentences,
                             num_keywords=num_keywords)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)

    params = list(lm_model.parameters())
    opt, sche = get_optimizer(lm_opt, lm_lr, decay=lm_decay,
                              train_step=rl_steps, params=params)

    global_step = 0

    stats = {
        "reward": []
    }

    for seq, mask, tok_type, label, lm_label, raw_text in tqdm.tqdm(dataloader, total=rl_steps):
        global_step += 1

        seq = seq.to(device)
        mask = mask.to(device)
        tok_type = tok_type.to(device)

        for looper in range(5):
            episode = []
            environment = Env(seq, mask, tok_type, raw_text, use_metric, dataset._tokenizer,
                              num_decode_iter, max_mask_rate, lm_model)

            for i in range(num_decode_iter):
                # print("iter", i)

                seq_input, pos = environment.get_state()
                # print("seq input", seq_input.size())
                # print("sent emb", environment.get_sentence_emb().shape)
                # print("tok type", tok_type.size())
                # print("mask size", mask.size())
                logits = lm_model(
                    sentence_embeds=environment.get_sentence_emb(),
                    input_ids=seq_input,
                    token_type_ids=tok_type,
                    attention_mask=mask)[0]

                seq_candidate = torch.multinomial(
                    F.softmax(logits, dim=2).view(-1, logits.size(2)), 1, replacement=False
                ).view_as(seq_input)
                logits_candidate = torch.gather(
                    F.log_softmax(logits, dim=2), dim=2, index=seq_candidate[:, :, None]).squeeze(2)

                # print("seq_candidate", seq_candidate.size())
                # print("logits_candidate", logits_candidate.size())
                reward = environment.step(seq_candidate, logits_candidate)
                episode.append([reward, torch.sum(logits_candidate * pos, dim=1)])


            stats["reward"].append(environment._score.copy())
            print("score", environment._score)
            print("stats", stats["reward"])
            print("mean", np.mean(stats["reward"]))

            episode[-1][0] -= np.mean(stats["reward"])
            for i in range(num_decode_iter - 2, -1, -1):
                episode[i][0] += discount * episode[i + 1][0]
            print(episode)

            loss = -sum([torch.mean(torch.tensor(reward).to(device) * logp) for reward, logp in episode])
            loss.backward()

        opt.step()
        sche.step()

        if global_step % lm_period_summary == 0:
            summary.add_scalar("rl/reward", np.mean(stats["reward"]), global_step)
            stats = {
                "reward": []
            }

        if global_step % lm_period_save == 0 or global_step == rl_steps:
            lm_model.to(torch.device("cpu")).eval()
            lm_model.save_pretrained(ckpt_path_pattern % (global_step // 1000))
            lm_model.to(device)

        if global_step >= rl_steps:
            break

    lm_model.eval()
    lm_model.to(torch.device("cpu"))
    return lm_model


def get_lm(lm_option, output_dir, trainset, device, lm_steps=5000, lm_bs=32,
           lm_opt="adamw", lm_lr=0.0001, lm_decay=0.01,
           lm_period_summary=100, lm_period_save=5000, **kwargs):
    """Returns a BERT language model or a list of language models on a given dataset.

    The language model will be stored at ``<output_dir>/lm_all`` if lm_option is finetune.
    The language model will be stored at ``<output_dir>/lm_filter_?`` if lm_option is adv.

    If filter is not -1. The pretrained language model will first be pretrained on the while
    dataset, then it will be finetuned on the data excluding the filter category.

    The re

    Args:
        lm_option (str): choose from `["pretrain", "finetune", "adv", "nartune"]`.
            pretrain means the pretrained BERT model without fine-tuning on current
            dataset.
            finetune means fine-tuning the BERT model on current dataset.
            adv means adversarial tuning on current dataset.
            nartune means tuning the
        output_dir (str): a directory to store pretrained language model.
        trainset (DatasetForBert): the training set for finetune the language model.
        device (torch.Device): a device to train the model.
        lm_steps (int): finetuning steps.
        lm_bs (int): finetuning batch size.
        lm_opt (str): optimzer name. choose from ["sgd", "adam", "adamW"].
        lm_lr (float): learning rate.
        lm_decay (float): weight decay for the optimizer.
        lm_period_summary (int): number of steps to write training summary.
        lm_period_save (int): number of steps to save the finetuned model.
    Returns:
        (BertTokenizerFast): the tokenizer for the language model.
        (BertForMaskedLM): a finetuned language model if lm_option is pretrain or finetune.
        ([BertForMaskedLM]): a list of finetuned language model if lm_option is adv. The i-th
            language model in the list is fine-tuned on data not having label i.
    """

    if trainset["cased"]:
        model_init = "bert-base-cased"
    else:
        model_init = "bert-base-uncased"

    tokenizer = BertTokenizerFast.from_pretrained(
        resources.get_transformers(model_init), do_lower_case="uncased" in model_init)
    tokenizer.do_lower_case = True if "uncased" in model_init else False

    if lm_option == "pretrain":
        bert_lm = BertForMaskedLM.from_pretrained(
            resources.get_transformers(model_init))
    elif lm_option == "finetune":
        bert_lm = fine_tune_lm(
            output_dir, trainset, -1, device,
            lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
            lm_period_summary=lm_period_summary, lm_period_save=lm_period_save)
    elif lm_option == "adv":
        bert_lm = []
        for i in range(len(trainset["label_mapping"])):
            lm = fine_tune_lm(
                output_dir, trainset, i, device,
                lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
                lm_period_summary=lm_period_summary, lm_period_save=lm_period_save)
            bert_lm.append(lm)
    elif lm_option == "nartune":
        bert_lm = non_autoregressive_fine_tune_lm(
            output_dir, trainset, -1, device, model_class=NonAutoregressiveBertLM,
            lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
            lm_period_summary=lm_period_summary, lm_period_save=lm_period_save, **kwargs)
    elif lm_option == "narrl":
        bert_lm = non_autoregressive_fine_tune_lm(
            output_dir, trainset, -1, device, model_class=NARRLBertLM,
            lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
            lm_period_summary=lm_period_summary, lm_period_save=lm_period_save, **kwargs)
        bert_lm = rl_fine_tune_lm(
            output_dir, trainset, -1, device, bert_lm,
            rl_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=0.00001,
            lm_decay=0, lm_period_summary=lm_period_summary, lm_period_save=lm_period_save,
            **kwargs)
    else:
        raise RuntimeError("unsupported lm_option")

    for model in (bert_lm if isinstance(bert_lm, list) else [bert_lm]):
        model.eval()
        for item in model.parameters():
            item.requires_grad = False

    return tokenizer, bert_lm
