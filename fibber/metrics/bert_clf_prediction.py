"""This metric outputs the BERT classifier prediction of the paraphrased text."""

import os

import numpy as np
import torch
import tqdm
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F

from fibber import get_root_dir, log
from fibber.datasets import DatasetForBert
from fibber.metrics.metric_base import MetricBase

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
        assert 0, "unkown optimizer"

    sche = transformers.get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup, num_training_steps=train_step)

    return opt, sche


def run_evaluate(model, dataloader_iter, eval_steps, summary, global_step, device):
    """Evaluate a model and add error rate and validation loss to Tensorboard.

    Args:
        model (transformers.BertForSequenceClassification): a BERT classification model.
        dataloader_iter (torch.IterableDataset): an iterator of a torch.IterableDataset.
        eval_steps (int): number of training steps.
        summary (torch.utils.tensorboard.SummaryWriter): a Tensorboard SummaryWriter object.
        global_step (int): current training steps.
        device (torch.Device): the device where the model in running on.
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
            outputs = model(seq, mask, tok_type, labels=label)
            loss, logits = outputs[:2]
            loss_list.append(loss.detach().cpu().numpy())
            count += seq.size(0)
            correct += (logits.argmax(dim=1).eq(label)
                        .float().sum().detach().cpu().numpy())

    summary.add_scalar("clf_val/error_rate", 1 - correct / count, global_step)
    summary.add_scalar("clf_val/loss", np.mean(loss_list), global_step)

    model.train()


def load_or_train_bert_clf(model_init,
                           dataset_name,
                           trainset,
                           testset,
                           bert_clf_steps,
                           bert_clf_bs,
                           bert_clf_lr,
                           bert_clf_optimizer,
                           bert_clf_weight_decay,
                           bert_clf_period_summary,
                           bert_clf_period_val,
                           bert_clf_period_save,
                           bert_clf_val_steps,
                           device):
    """Train BERT classification model on a dataset.

    The trained model will be stored at ``<fibber_root_dir>/bert_clf/<dataset_name>/``. If there's
    a saved model, load and return the model. Otherwise, train the model using the given data.

    Args:
        model_init (str): pretrained model name. Choose from ``["bert-base-cased",
            "bert-base-uncased", "bert-large-cased", "bert-large-uncased"]``.
        dataset_name (str): the name of the dataset. This is also the dir to save trained model.
        trainset (dict): a fibber dataset.
        testset (dict): a fibber dataset.
        bert_clf_steps (int): steps to train a classifier.
        bert_clf_bs (int): the batch size.
        bert_clf_lr (float): the learning rate.
        bert_clf_optimizer (str): the optimizer name.
        bert_clf_weight_decay (float): the weight decay.
        bert_clf_period_summary (int): the period in steps to write training summary.
        bert_clf_period_val (int): the period in steps to run validation and write validation
            summary.
        bert_clf_period_save (int): the period in steps to save current model.
        bert_clf_val_steps (int): number of batched in each validation.
        device (torch.Device): the device to run the model.

    Returns:
        (transformers.BertForSequenceClassification): a torch BERT model.
    """
    model_dir = os.path.join(get_root_dir(), "bert_clf", dataset_name, )
    ckpt_path = os.path.join(model_dir, model_init + "-%04dk.pt" %
                             (bert_clf_steps // 1000))

    if os.path.exists(ckpt_path):
        logger.info("Load BERT classifier from %s.", ckpt_path)
        model = torch.load(ckpt_path)
        model.eval()
        model.to(device)
        return model

    num_labels = len(trainset["label_mapping"])

    model = BertForSequenceClassification.from_pretrained(
        model_init, num_labels=num_labels).to(device)
    model.train()

    logger.info("Use %s tokenizer and classifier.", model_init)
    logger.info("Num labels: %s", num_labels)

    summary = SummaryWriter(os.path.join(model_dir, "summary"))

    dataloader = torch.utils.data.DataLoader(
        DatasetForBert(trainset, model_init, bert_clf_bs), batch_size=None, num_workers=2)

    dataloader_val = torch.utils.data.DataLoader(
        DatasetForBert(testset, model_init, bert_clf_bs), batch_size=None, num_workers=1)
    dataloader_val_iter = iter(dataloader_val)

    params = model.parameters()

    opt, sche = get_optimizer(
        bert_clf_optimizer, bert_clf_lr, bert_clf_weight_decay, bert_clf_steps, params)

    global_step = 0
    correct_train, count_train = 0, 0
    for seq, mask, tok_type, label in tqdm.tqdm(dataloader, total=bert_clf_steps):
        global_step += 1
        seq = seq.to(device)
        mask = mask.to(device)
        tok_type = tok_type.to(device)
        label = label.to(device)

        outputs = model(seq, mask, tok_type, labels=label)
        loss, logits = outputs[:2]

        count_train += seq.size(0)
        correct_train += (logits.argmax(dim=1).eq(label)
                          .float().sum().detach().cpu().numpy())

        opt.zero_grad()
        loss.backward()
        opt.step()
        sche.step()

        if global_step % bert_clf_period_summary == 0:
            summary.add_scalar("clf_train/loss", loss, global_step)
            summary.add_scalar("clf_train/error_rate", 1 - correct_train / count_train,
                               global_step)
            correct_train, count_train = 0, 0

        if global_step % bert_clf_period_val == 0:
            run_evaluate(model, dataloader_val_iter,
                         bert_clf_val_steps, summary, global_step, device)

        if global_step % bert_clf_period_save == 0 or global_step == bert_clf_steps:
            model.eval()
            model.to(torch.device("cpu"))
            ckpt = os.path.join(model_dir, model_init + "-%04dk.pt" % (bert_clf_steps // 1000))
            torch.save(model, ckpt)
            logger.info("BERT classifier saved at %s.", ckpt)
            model.to(device)
            model.train()

        if global_step >= bert_clf_steps:
            break
    model.eval()
    return model


class BertClfPrediction(MetricBase):
    """BERT classifier prediction on paraphrases.

    This metric is special, it does not compare the original and paraphrased sentence.
    Instead, it outputs the classifier prediction on paraphrases. So we should not compute
    mean or std on this metric.

    Args:
        dataset_name (str): the name of the dataset.
        trainset (dict): a fibber dataset.
        testset (dict): a fibber dataset.
        bert_gpu_id (int): the gpu id for BERT model. Set -1 to use CPU.
        bert_clf_steps (int): steps to train a classifier.
        bert_clf_bs (int): the batch size.
        bert_clf_lr (float): the learning rate.
        bert_clf_optimizer (str): the optimizer name.
        bert_clf_weight_decay (float): the weight decay in the optimizer.
        bert_clf_period_summary (int): the period in steps to write training summary.
        bert_clf_period_val (int): the period in steps to run validation and write validation
            summary.
        bert_clf_period_save (int): the period in steps to save current model.
        bert_clf_val_steps (int): number of batched in each validation.
    """

    def __init__(self, dataset_name, trainset, testset, bert_gpu_id=-1,
                 bert_clf_steps=20000, bert_clf_bs=32, bert_clf_lr=0.00002,
                 bert_clf_optimizer="adamw", bert_clf_weight_decay=0.001,
                 bert_clf_period_summary=100, bert_clf_period_val=500,
                 bert_clf_period_save=5000, bert_clf_val_steps=10, **kargs):

        super(BertClfPrediction, self).__init__()

        if trainset["cased"]:
            model_init = "bert-base-cased"
            logger.info(
                "Use cased model in BERT classifier prediction metric.")
        else:
            model_init = "bert-base-uncased"
            logger.info(
                "Use uncased model in BERT classifier prediction metric.")

        self._tokenizer = BertTokenizer.from_pretrained(model_init)
        if bert_gpu_id == -1:
            logger.warning("BERT metric is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("BERT metric is running on GPU %d.", bert_gpu_id)
            self._device = torch.device("cuda:%d" % bert_gpu_id)

        self._model = load_or_train_bert_clf(
            model_init=model_init,
            dataset_name=dataset_name,
            trainset=trainset,
            testset=testset,
            bert_clf_steps=bert_clf_steps,
            bert_clf_lr=bert_clf_lr,
            bert_clf_bs=bert_clf_bs,
            bert_clf_optimizer=bert_clf_optimizer,
            bert_clf_weight_decay=bert_clf_weight_decay,
            bert_clf_period_summary=bert_clf_period_summary,
            bert_clf_period_val=bert_clf_period_val,
            bert_clf_period_save=bert_clf_period_save,
            bert_clf_val_steps=bert_clf_val_steps,
            device=self._device)

    def predict_raw(self, text0, text1):
        """Get unnormalized log probability of the BERT prediction.

        Args:
            text0 (str):
                On regular classification datasets, text0 contains the text to be classified.
                On NLI datasets, text0 contains the premise.
            text1 (str or None):
                On regular classification datasets, text0 is None.
                On NLI datasets, text1 contains the hypothesis.

        Returns:
            (np.array): a numpy array of the unnormalized log probability over on each category.
        """
        if text1 is None:
            seq = ["[CLS]"] + self._tokenizer.tokenize(text0) + ["[SEP]"]
            seq_tensor = torch.tensor(self._tokenizer.convert_tokens_to_ids(seq)).to(self._device)
            seq_tensor = seq_tensor[:200]
            return F.log_softmax(self._model(seq_tensor.unsqueeze(0))[0],
                                 dim=-1)[0].detach().cpu().numpy()
        else:
            seq0 = self._tokenizer.tokenize(text0) + ["[SEP]"]
            seq1 = self._tokenizer.tokenize(text1) + ["[SEP]"]
            l0 = len(seq0)
            l1 = len(seq1)
            seq_tensor = torch.tensor(
                self._tokenizer.convert_tokens_to_ids(["[CLS]"] + seq0 + seq1)
            ).to(self._device).unsqueeze(0)
            tok_type = torch.tensor(
                [0] * (l0 + 1) + [1] * l1).to(self._device).unsqueeze(0)
            seq_tensor = seq_tensor[:, :200]
            tok_type = tok_type[:, :200]
            return F.log_softmax(self._model(seq_tensor, token_type_ids=tok_type)[0],
                                 dim=-1)[0].detach().cpu().numpy()


    def predict_raw_batch(self, text0, text1):
        if text1 is None:
            seq = [["[CLS]"] + self._tokenizer.tokenize(x)[:200] + ["[SEP]"] for x in text0]
            seq_len = [len(x) for x in seq]
            max_len = max(seq_len)
            seq = [x + ["[SEQ]"] * (max_len - y) for x, y in zip(seq, seq_len)]

            mask = [[1] * x + [0] * (max_len - x) for x in seq_len]

            seq_tensor = torch.tensor(
                [self._tokenizer.convert_tokens_to_ids(x) for x in seq]).to(self._device)
            mask_tensor = torch.tensor(mask).to(self._device)

            return F.log_softmax(self._model(seq_tensor, mask_tensor)[0], dim=1).detach().cpu().numpy()
        else:
            assert 0

    def predict(self, text0, text1):
        """Get the prediction of the BERT prediction.

        Args:
            text0 (str):
                On regular classification datasets, text0 contains the text to be classified.
                On NLI datasets, text0 contains the premise.
            text1 (str or None):
                On regular classification datasets, text0 is None.
                On NLI datasets, text1 contains the hypothesis.

        Returns:
            (int): the prediction of the classifier.
        """

        return int(np.argmax(self.predict_raw(text0, text1)))

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        if paraphrase_field == "text0":
            return self.predict(paraphrase, None)
        else:
            assert paraphrase_field == "text1"
            return self.predict(data_record["text0"], paraphrase)
