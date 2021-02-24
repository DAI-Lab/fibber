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

logger = log.setup_custom_logger(__name__)


class NonAutoregressiveBertLM(BertForMaskedLM):
    def __init__(self, config, sentence_embed_size):
        super().__init__(config)
        self.sentence_emb_transform = nn.Linear(sentence_embed_size, self.config.hidden_size)

    def _compute_emb(self, sentence_embeds, input_ids):
        # print(sentence_embeds.shape)
        transformed_sentence_embs = self.sentence_emb_transform(
            torch.tensor(sentence_embeds).to(self.device))
        bert_input_embs = self.bert.embeddings.word_embeddings(input_ids)
        inp_emb = torch.cat([transformed_sentence_embs[:, None, :], bert_input_embs[:, 1:]], dim=1)

        return inp_emb

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
        output_dir, trainset, filter, device, use_metric, lm_steps=5000, lm_bs=32,
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
        use_metric (USESemanticSimilarityMetric): a sentence encoder metric
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
        output_dir_t = os.path.join(output_dir, "narlm_all")
    else:
        output_dir_t = os.path.join(output_dir, "narlm_filter_%d" % filter)

    summary = SummaryWriter(output_dir_t + "/summary")

    if trainset["cased"]:
        model_init = "bert-base-cased"
    else:
        model_init = "bert-base-uncased"

    ckpt_path_pattern = output_dir_t + "/checkpoint-%04dk"
    ckpt_path = ckpt_path_pattern % (lm_steps // 1000)

    if os.path.exists(ckpt_path):
        logger.info("Language model <%s> exists.", ckpt_path)
        return NonAutoregressiveBertLM.from_pretrained(ckpt_path, sentence_embed_size=512).eval()

    if filter == -1:
        lm_model = NonAutoregressiveBertLM.from_pretrained(resources.get_transformers(model_init),
                                                           sentence_embed_size=512)
        lm_model.train()
    else:
        lm_model = get_lm(output_dir, trainset, -1, device, lm_steps)
        lm_model.train()
    lm_model.to(device)

    dataset = DatasetForBert(trainset, model_init, lm_bs, exclude=filter, masked_lm=False,
                             dynamic_masked_lm=True, include_raw_text=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=2)

    params = list(lm_model.parameters())
    opt, sche = get_optimizer(lm_opt, lm_lr, lm_decay, lm_steps, params)

    global_step = 0
    stats = new_stats()
    for seq, mask, tok_type, label, lm_label, raw_text in tqdm.tqdm(
            dataloader, total=lm_steps):
        opt.zero_grad()

        global_step += 1

        raw_text = [x.replace("[CLS]", "").replace("[SEP]", "") for x in raw_text]
        sentence_embs = use_metric.model(raw_text).numpy()

        seq = seq.to(device)
        mask = mask.to(device)
        tok_type = tok_type.to(device)
        label = label.to(device)
        lm_label = lm_label.to(device)

        # print("seq", seq.size())
        # print("mask", mask.size())
        # print("toktype", tok_type.size())
        # print("label", label.size())
        # print("lmlabel", lm_label.size())

        lm_loss = compute_non_autoregressive_lm_loss(
            lm_model=lm_model, sentence_embeds=sentence_embs, seq=seq, mask=mask,
            tok_type=tok_type, lm_label=lm_label, stats=stats)
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
        bert_lm.eval()
        for item in bert_lm.parameters():
            item.requires_grad = False

    elif lm_option == "finetune":
        bert_lm = fine_tune_lm(
            output_dir, trainset, -1, device,
            lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
            lm_period_summary=lm_period_summary, lm_period_save=lm_period_save)
        bert_lm.eval()
        for item in bert_lm.parameters():
            item.requires_grad = False

    elif lm_option == "adv":
        bert_lm = []
        for i in range(len(trainset["label_mapping"])):
            lm = fine_tune_lm(
                output_dir, trainset, i, device,
                lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
                lm_period_summary=lm_period_summary, lm_period_save=lm_period_save)
            lm.eval()
            for item in lm.parameters():
                item.requires_grad = False
            bert_lm.append(lm)

    elif lm_option == "nartune":
        bert_lm = non_autoregressive_fine_tune_lm(
            output_dir, trainset, -1, device,
            lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
            lm_period_summary=lm_period_summary, lm_period_save=lm_period_save, **kwargs)
        bert_lm.eval()
        for item in bert_lm.parameters():
            item.requires_grad = False
    else:
        raise RuntimeError("unsupported lm_option")

    return tokenizer, bert_lm
