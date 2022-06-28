"""This metric outputs a transformer-based classifier prediction of the paraphrased text."""

import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fibber import get_root_dir, log, resources
from fibber.datasets import DatasetForTransformers
from fibber.metrics.classifier.classifier_base import ClassifierBase

logger = log.setup_custom_logger(__name__)


def get_optimizer(optimizer_name, lr, decay, train_step, params, warmup=1000):
    """Create an optimizer and schedule of learning rate for parameters.

    Args:
        optimizer_name (str): choose from ``["adam", "sgd", "adamw"]``.
        lr (float): learning rate.
        decay (float): weight decay.
        train_step (int): number of training steps.
        params (list): a list of parameters in the model.
        warmup (int): number of warm up steps.

    Returns:
        A torch optimizer and a scheduler.
    """
    if optimizer_name == "adam":
        opt = torch.optim.Adam(params=params, lr=lr, weight_decay=decay)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(params=params, lr=lr, weight_decay=decay)
    elif optimizer_name == "adamw":
        opt = torch.optim.AdamW(params=params, lr=lr, weight_decay=decay)
    else:
        raise RuntimeError("unknown optimizer")

    schedule = transformers.get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup, num_training_steps=train_step)

    return opt, schedule


def run_evaluate(model, dataloader_iter, eval_steps, summary, global_step, device, model_init):
    """Evaluate a model and add error rate and validation loss to Tensorboard.

    Args:
        model (transformers.BertForSequenceClassification): a BERT classification model.
        dataloader_iter (torch.IterableDataset): an iterator of a torch.IterableDataset.
        eval_steps (int): number of training steps.
        summary (torch.utils.tensorboard.SummaryWriter): a Tensorboard SummaryWriter object.
        global_step (int): current training steps.
        device (torch.Device): the device where the model in running on.
        model_init (str): a str specifies the pretrained model. used to determine model input.
    """
    model.eval()

    correct, count = 0, 0
    loss_list = []

    for v_step in range(eval_steps):
        seq, mask, tok_type, label = next(dataloader_iter)
        seq = seq.to(device)
        mask = mask.to(device)
        tok_type = tok_type.to(device)
        label = label.to(device)

        with torch.no_grad():
            if model_init.startswith("bert-"):
                outputs = model(seq, mask, tok_type, labels=label)
            else:
                outputs = model(seq, mask, labels=label)
            loss, logits = outputs[:2]
            loss_list.append(loss.detach().cpu().numpy())
            count += seq.size(0)
            correct += (logits.argmax(dim=1).eq(label)
                        .float().sum().detach().cpu().numpy())

    summary.add_scalar("clf_val/error_rate", 1 - correct / count, global_step)
    summary.add_scalar("clf_val/loss", np.mean(loss_list), global_step)

    model.train()


def load_or_train_transformer_clf(
        model_init, dataset_name, trainset, testset,
        transformer_clf_steps, transformer_clf_bs, transformer_clf_lr, transformer_clf_optimizer,
        transformer_clf_weight_decay, transformer_clf_period_summary, transformer_clf_period_val,
        transformer_clf_period_save, transformer_clf_val_steps, device):
    """Train transformer-based classification model on a dataset.

    The trained model will be stored at ``<fibber_root_dir>/transformer_clf/<dataset_name>/``.
    If there's a saved model, load and return the model. Otherwise, train the model using the
    given data.

    Args:
        model_init (str): pretrained model name. e.g. ``["bert-base-cased",
            "bert-base-uncased", "bert-large-cased", "bert-large-uncased", "roberta-large"]``.
        dataset_name (str): the name of the dataset. This is also the dir to save trained model.
        trainset (dict): a fibber dataset.
        testset (dict): a fibber dataset.
        transformer_clf_steps (int): steps to train a classifier.
        transformer_clf_bs (int): the batch size.
        transformer_clf_lr (float): the learning rate.
        transformer_clf_optimizer (str): the optimizer name.
        transformer_clf_weight_decay (float): the weight decay.
        transformer_clf_period_summary (int): the period in steps to write training summary.
        transformer_clf_period_val (int): the period in steps to run validation and write
            validation summary.
        transformer_clf_period_save (int): the period in steps to save current model.
        transformer_clf_val_steps (int): number of batched in each validation.
        device (torch.Device): the device to run the model.

    Returns:
        a torch transformer model.
    """
    model_dir = os.path.join(get_root_dir(), "transformer_clf", dataset_name)
    ckpt_path = os.path.join(model_dir, model_init + "-%04dk" %
                             (transformer_clf_steps // 1000))

    if os.path.exists(ckpt_path):
        logger.info("Load transformer classifier from %s.", ckpt_path)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        model.eval()
        model.to(device)
        return model

    num_labels = len(trainset["label_mapping"])

    model = AutoModelForSequenceClassification.from_pretrained(
        resources.get_transformers(model_init), num_labels=num_labels).to(device)
    model.train()

    logger.info("Use %s tokenizer and classifier.", model_init)
    logger.info("Num labels: %s", num_labels)

    summary = SummaryWriter(os.path.join(model_dir, "summary"))

    dataloader = torch.utils.data.DataLoader(
        DatasetForTransformers(trainset, model_init, transformer_clf_bs), batch_size=None,
        num_workers=0)

    dataloader_val = torch.utils.data.DataLoader(
        DatasetForTransformers(testset, model_init, transformer_clf_bs), batch_size=None,
        num_workers=0)
    dataloader_val_iter = iter(dataloader_val)

    params = model.parameters()

    opt, schedule = get_optimizer(
        transformer_clf_optimizer, transformer_clf_lr, transformer_clf_weight_decay,
        transformer_clf_steps, params)

    global_step = 0
    correct_train, count_train = 0, 0
    for seq, mask, tok_type, label in tqdm.tqdm(dataloader, total=transformer_clf_steps):
        global_step += 1
        seq = seq.to(device)
        mask = mask.to(device)
        tok_type = tok_type.to(device)
        label = label.to(device)

        if model_init.startswith("bert-"):
            outputs = model(input_ids=seq, attention_mask=mask,
                            token_type_ids=tok_type, labels=label)
        else:
            outputs = model(input_ids=seq, attention_mask=mask, labels=label)

        loss, logits = outputs[:2]

        count_train += seq.size(0)
        correct_train += (logits.argmax(dim=1).eq(label)
                          .float().sum().detach().cpu().numpy())

        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()

        if global_step % transformer_clf_period_summary == 0:
            summary.add_scalar("clf_train/loss", loss, global_step)
            summary.add_scalar("clf_train/error_rate", 1 - correct_train / count_train,
                               global_step)
            correct_train, count_train = 0, 0

        if global_step % transformer_clf_period_val == 0:
            run_evaluate(model, dataloader_val_iter,
                         transformer_clf_val_steps, summary, global_step, device, model_init)

        if global_step % transformer_clf_period_save == 0 or global_step == transformer_clf_steps:
            ckpt_path = os.path.join(model_dir, model_init + "-%04dk" % (global_step // 1000))
            model.save_pretrained(ckpt_path)
            logger.info("transformer classifier saved at %s.", ckpt_path)

        if global_step >= transformer_clf_steps:
            break
    model.eval()
    return model


def trivial_filtering(sents, tokenizer, vocab):
    tokens = [tokenizer.tokenize(sent) for sent in sents]
    for i in range(len(sents)):
        trivial_wid = [vocab[tok] if tok in vocab else 1e9 for tok in tokens[i]]
        pos = np.argmin(trivial_wid)
        tokens[i][pos] = "[MASK]"
    return [tokenizer.convert_tokens_to_string(row) for row in tokens]


class TransformerClassifier(ClassifierBase):
    """BERT classifier prediction on paraphrase_list.

    This metric is special, it does not compare the original and paraphrased sentence.
    Instead, it outputs the classifier prediction on paraphrase_list. So we should not compute
    mean or std on this metric.

    Args:
        dataset_name (str): the name of the dataset.
        trainset (dict): a fibber dataset.
        testset (dict): a fibber dataset.
        transformer_clf_gpu_id (int): the gpu id for BERT model. Set -1 to use CPU.
        transformer_clf_steps (int): steps to train a classifier.
        transformer_clf_bs (int): the batch size.
        transformer_clf_lr (float): the learning rate.
        transformer_clf_optimizer (str): the optimizer name.
        transformer_clf_weight_decay (float): the weight decay in the optimizer.
        transformer_clf_period_summary (int): the period in steps to write training summary.
        transformer_clf_period_val (int): the period in steps to run validation and write
            validation summary.
        transformer_clf_period_save (int): the period in steps to save current model.
        transformer_clf_val_steps (int): number of batched in each validation.
    """

    def __init__(self, dataset_name, trainset, testset, transformer_clf_gpu_id=-1,
                 transformer_clf_steps=20000, transformer_clf_bs=32, transformer_clf_lr=0.00002,
                 transformer_clf_optimizer="adamw", transformer_clf_weight_decay=0.001,
                 transformer_clf_period_summary=100, transformer_clf_period_val=500,
                 transformer_clf_period_save=20000, transformer_clf_val_steps=10,
                 transformer_clf_model_init="bert-base-cased", **kwargs):
        super(TransformerClassifier, self).__init__(**kwargs)
        logger.info("Use %s classifier.", transformer_clf_model_init)

        self._tokenizer = AutoTokenizer.from_pretrained(
            resources.get_transformers(transformer_clf_model_init))

        if transformer_clf_gpu_id == -1:
            logger.warning("Transformer clf metric is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("Transformer clf metric is running on GPU %d.", transformer_clf_gpu_id)
            self._device = torch.device("cuda:%d" % transformer_clf_gpu_id)

        self._model_init = transformer_clf_model_init
        self._dataset_name = dataset_name
        self._model = load_or_train_transformer_clf(
            model_init=transformer_clf_model_init,
            dataset_name=dataset_name,
            trainset=trainset,
            testset=testset,
            transformer_clf_steps=transformer_clf_steps,
            transformer_clf_lr=transformer_clf_lr,
            transformer_clf_bs=transformer_clf_bs,
            transformer_clf_optimizer=transformer_clf_optimizer,
            transformer_clf_weight_decay=transformer_clf_weight_decay,
            transformer_clf_period_summary=transformer_clf_period_summary,
            transformer_clf_period_val=transformer_clf_period_val,
            transformer_clf_period_save=transformer_clf_period_save,
            transformer_clf_val_steps=transformer_clf_val_steps,
            device=self._device)
        self._fine_tune_schedule = None
        self._fine_tune_opt = None
        self._ppl_filter_metric = None
        self._num_labels = len(trainset["label_mapping"])

    def __repr__(self):
        return self._model_init + "-Classifier"

    def get_model_and_tokenizer(self):
        return self._model, self._tokenizer

    def get_model_init(self):
        return self._model_init

    def get_device(self):
        return self._device

    def enable_ppl_filter(self, ppl_metric):
        self._ppl_filter_metric = ppl_metric

    def _predict_log_dist_batch(self, origin, paraphrase_list, data_record=None):
        """Predict the log-probability distribution over classes for one batch.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.

        Returns:
            (np.array): a numpy array of size ``(batch_size * num_labels)``.
        """
        if self._ppl_filter_metric is not None:
            paraphrase_list = self._ppl_filter_metric.perplexity_filter(paraphrase_list)

        if self._field == "text0":
            batch_input = self._tokenizer(
                text=paraphrase_list, padding=True, return_tensors="pt").to(self._device)
        else:
            assert self._field == "text1"
            batch_input = self._tokenizer(text=[data_record["text0"]] * len(paraphrase_list),
                                          text_pair=paraphrase_list,
                                          padding=True, return_tensors="pt").to(self._device)
        with torch.no_grad():
            logits = self._model(**batch_input)[0]
            res = F.log_softmax(logits, dim=1).detach().cpu().numpy()

        return res

    def _predict_log_dist_multiple_examples(self, origin_list, paraphrase_list,
                                            data_record_list=None, return_raw_logits=False):
        if self._ppl_filter_metric is not None:
            paraphrase_list = self._ppl_filter_metric.perplexity_filter(paraphrase_list)

        if self._field == "text0":
            batch_input = self._tokenizer(
                text=paraphrase_list, padding=True, return_tensors="pt").to(self._device)
        else:
            assert self._field == "text1"
            batch_input = self._tokenizer(text=[item["text0"] for item in data_record_list],
                                          text_pair=paraphrase_list,
                                          padding=True, return_tensors="pt").to(self._device)
        with torch.no_grad():
            logits = self._model(**batch_input)[0]
            if not return_raw_logits:
                res = F.log_softmax(logits, dim=1).detach().cpu().numpy()
            else:
                res = logits.detach().cpu().numpy()
        return res

    def _predict_log_dist_example(self, origin, paraphrase, data_record=None):
        """Predict the log-probability distribution over classes for one example.

        Args:
            origin (str): the original text.
            paraphrase (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.

        Returns:
            (np.array): a numpy array of size ``(num_labels)``.
        """
        return self.predict_log_dist_batch(origin, [paraphrase], data_record)[0]

    def robust_tune_init(self, optimizer, lr, weight_decay, steps):
        if self._fine_tune_schedule is not None or self._fine_tune_opt is not None:
            logger.error("fine tuning has been initialized.")
            raise RuntimeError("fine tuning has been initialized.")

        params = self._model.parameters()
        self._fine_tune_opt, self._fine_tune_schedule = get_optimizer(
            optimizer, lr, weight_decay, steps, params)

    def robust_tune_step(self, data_record_list):
        if self._fine_tune_schedule is None or self._fine_tune_opt is None:
            logger.error("fine tuning not initialized.")
            raise RuntimeError("fine tuning not initialized.")

        self._model.train()
        text0_list = []
        text1_list = []
        labels = []
        for data_record in data_record_list:
            if "text0" in data_record:
                text0_list.append(data_record["text0"])
            if "text1" in data_record:
                text1_list.append(data_record["text1"])
            labels.append(data_record["label"])

        if len(text1_list) == 0:
            text1_list = None
        elif len(text1_list) != len(text0_list):
            raise RuntimeError("data records are not consistent.")

        batch_input = self._tokenizer(text=text0_list, text_pair=text1_list, padding=True,
                                      return_tensors="pt").to(self._device)

        loss, logits = self._model(
            labels=torch.tensor(labels).to(self._device),
            **batch_input
        )[:2]

        self._fine_tune_opt.zero_grad()
        loss.backward()
        self._fine_tune_opt.step()
        self._fine_tune_schedule.step()

        self._model.eval()
        return logits.argmax(axis=1).detach().cpu().numpy(), float(loss.detach().cpu().numpy())

    def load_robust_tuned_model(self, load_path):
        model_dir = os.path.join(load_path, "model_ckpt")
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        self._model.eval()
        self._model.to(self._device)
        logger.info("Load transformer-based classifier from %s.", model_dir)

    def save_robust_tuned_model(self, save_path):
        model_dir = os.path.join(save_path, "model_ckpt")
        self._model.save_pretrained(model_dir)
        logger.info("Transformer-based classifier saved at %s.", model_dir)
