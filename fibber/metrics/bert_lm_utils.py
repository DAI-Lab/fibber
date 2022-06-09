import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, BertLMHeadModel

from fibber import get_root_dir, log, resources
from fibber.datasets import DatasetForTransformers
from fibber.metrics.classifier.transformer_classifier import get_optimizer

logger = log.setup_custom_logger(__name__)


def write_summary(stats, summary, global_step):
    """Save langauge model training summary."""
    summary.add_scalar("loss_lm", np.mean(stats["lm_loss"]), global_step)
    summary.add_scalar("error_lm", 1 - stats["lm_correct"] / stats["lm_total"],
                       global_step)


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


def fine_tune_lm(output_dir, trainset, filter, device, model_init="bert-base-cased",
                 lm_steps=5000, lm_bs=32, lm_opt="adamw", lm_lr=0.0001, lm_decay=0.01,
                 lm_period_summary=100, lm_period_save=5000, as_masked_lm=True,
                 select_field=None):
    """Returns a finetuned BERT language model on a given dataset.

    The language model will be stored at ``<output_dir>/lm_all`` if filter is -1, or
    ``<output_dir>/lm_filter_?`` if filter is not -1.

    If filter is not -1. The pretrained langauge model will first be pretrained on the while
    dataset, then it will be finetuned on the data excluding the filter category.

    Args:
        output_dir (str): a directory to store pretrained language model.
        trainset (DatasetForTransformers): the training set for finetune the language model.
        filter (int): a category to exclude from finetuning.
        device (torch.Device): a device to train the model.
        model_init (str): the backbone bert model.
        lm_steps (int): finetuning steps.
        lm_bs (int): finetuning batch size.
        lm_opt (str): optimzer name. choose from ["sgd", "adam", "adamW"].
        lm_lr (float): learning rate.
        lm_decay (float): weight decay for the optimizer.
        lm_period_summary (int): number of steps to write training summary.
        lm_period_save (int): number of steps to save the finetuned model.
        as_masked_lm (bool): use BERT as a masked language model. If False, use as auto-regressive.
        select_field (None or str): select one field for language model.
    Returns:
        (BertForMaskedLM): a finetuned language model.
    """

    if as_masked_lm:
        if filter == -1:
            folder = "masked_lm_all"
        else:
            folder = "masked_lm_filter_%d" % filter
    else:
        if filter == -1:
            folder = "autoregressive_lm_all"
        else:
            folder = "autoregressive_lm_filter_%d" % filter

    output_dir_t = os.path.join(output_dir, folder)

    ckpt_path_pattern = output_dir_t + "/checkpoint-%04dk"
    ckpt_path = ckpt_path_pattern % (lm_steps // 1000)

    if os.path.exists(ckpt_path):
        logger.info("Language model <%s> exists.", ckpt_path)
        if as_masked_lm:
            return BertForMaskedLM.from_pretrained(ckpt_path).eval()
        else:
            config = BertConfig.from_pretrained(resources.get_transformers(model_init))
            config.is_decoder = True
            return BertLMHeadModel.from_pretrained(ckpt_path, config=config).eval()

    summary = SummaryWriter(output_dir_t + "/summary")
    if as_masked_lm:
        dataset = DatasetForTransformers(trainset, model_init, lm_bs, exclude=filter,
                                         masked_lm=True, select_field=select_field)
        lm_model = BertForMaskedLM.from_pretrained(resources.get_transformers(model_init))
    else:
        dataset = DatasetForTransformers(trainset, model_init, lm_bs, exclude=filter,
                                         autoregressive_lm=True, select_field=select_field)
        config = BertConfig.from_pretrained(resources.get_transformers(model_init))
        config.is_decoder = True
        lm_model = BertLMHeadModel.from_pretrained(resources.get_transformers(model_init),
                                                   config=config)
    lm_model.train()
    lm_model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)

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
        lm_label = lm_label.to(device)

        lm_loss = compute_lm_loss(lm_model, seq, mask, tok_type, lm_label, stats)
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


def get_lm(lm_option, dataset_name, trainset, device, model_init="bert-base-cased",
           filter=-1, lm_steps=5000, lm_bs=32, lm_opt="adamw", lm_lr=0.0001, lm_decay=0.01,
           lm_period_summary=100, lm_period_save=5000, select_field=None):
    """Returns a BERT language model or a list of language models on a given dataset.

    The language model will be stored at ``<output_dir>/lm_all`` if lm_option is finetune.
    The language model will be stored at ``<output_dir>/lm_filter_?`` if lm_option is adv.

    If filter is not -1. The pretrained language model will first be pretrained on the while
    dataset, then it will be finetuned on the data excluding the filter category.

    The re

    Args:
        lm_option (str): choose from `["pretrain", "finetune", "adv"]`.
            pretrain means the pretrained BERT model without fine-tuning on current
            dataset.
            finetune means fine-tuning the BERT model on current dataset.
            adv means adversarial tuning on current dataset.
        dataset_name (str): a directory to store pretrained language model.
        trainset (dict): the training set for finetune the language model.
        device (torch.Device): a device to train the model.
        model_init (str): the backbone bert model.
        lm_steps (int): finetuning steps.
        lm_bs (int): finetuning batch size.
        lm_opt (str): optimzer name. choose from ["sgd", "adam", "adamW"].
        lm_lr (float): learning rate.
        lm_decay (float): weight decay for the optimizer.
        lm_period_summary (int): number of steps to write training summary.
        lm_period_save (int): number of steps to save the finetuned model.
        select_field (str or None): train language model on one specific field.
    Returns:
        (BertTokenizerFast): the tokenizer for the language model.
        (BertForMaskedLM): a finetuned language model if lm_option is pretrain or finetune.
        ([BertForMaskedLM]): a list of finetuned language model if lm_option is adv. The i-th
            language model in the list is fine-tuned on data not having label i.
    """
    tokenizer = AutoTokenizer.from_pretrained(resources.get_transformers(model_init))

    output_dir = os.path.join(get_root_dir(), "bert_lm", dataset_name)

    if lm_option == "pretrain":
        assert filter == -1
        bert_lm = BertForMaskedLM.from_pretrained(
            resources.get_transformers(model_init))
    elif lm_option == "finetune":
        bert_lm = fine_tune_lm(
            output_dir, trainset, filter, device, model_init=model_init,
            lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
            lm_period_summary=lm_period_summary, lm_period_save=lm_period_save,
            select_field=select_field)
    elif lm_option == "adv":
        bert_lm = []
        assert filter == -1
        for i in range(len(trainset["label_mapping"])):
            lm = fine_tune_lm(
                output_dir, trainset, i, device, model_init=model_init,
                lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
                lm_period_summary=lm_period_summary, lm_period_save=lm_period_save,
                select_field=select_field)
            bert_lm.append(lm)
    elif lm_option == "ppl":
        bert_lm = fine_tune_lm(
            output_dir, trainset, filter, device, model_init=model_init,
            lm_steps=lm_steps, lm_bs=lm_bs, lm_opt=lm_opt, lm_lr=lm_lr, lm_decay=lm_decay,
            lm_period_summary=lm_period_summary, lm_period_save=lm_period_save, as_masked_lm=False,
            select_field=select_field)
    else:
        raise RuntimeError("unsupported lm_option")

    for model in (bert_lm if isinstance(bert_lm, list) else [bert_lm]):
        model.eval()
        for item in model.parameters():
            item.requires_grad = False

    return tokenizer, bert_lm
