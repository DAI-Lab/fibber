import os

import numpy as np
import torch
import tqdm
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification, BertTokenizer

from .. import log
from ..dataset.dataset_utils import DatasetForBert
from ..download_utils import get_root_dir
from .measurement_base import MeasurementBase

logger = log.setup_custom_logger(__name__)


def get_optimizer(optim, lr, decay, train_step, params, warmup=500):
    if optim == "adam":
        opt = torch.optim.Adam(params=params, lr=lr, weight_decay=decay)
    elif optim == "sgd":
        opt = torch.optim.SGD(params=params, lr=lr, weight_decay=decay)
    elif optim == "adamw":
        opt = torch.optim.AdamW(params=params, lr=lr, weight_decay=decay)
    else:
        assert 0, "unkown optimizer"

    sche = transformers.get_linear_schedule_with_warmup(
        opt, num_warmup_steps=1000, num_training_steps=train_step)

    return opt, sche


def run_evaluate(model, dataloader_iter, eval_steps, summary, global_step, device):
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


def load_or_train_bert_clf(tokenizer,
                           model_init,
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
    model_dir = os.path.join(get_root_dir(), "bert_clf", dataset_name, )
    ckpt_path = os.path.join(model_dir, model_init + "-%04dk.pt" %
                             (bert_clf_steps // 1000))

    if os.path.exists(ckpt_path):
        logger.info("Load bert classifier from %s.", ckpt_path)
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

        if global_step % bert_clf_period_save == 0:
            model.eval()
            model.to(torch.device("cpu"))
            ckpt = os.path.join(model_dir, model_init + "-%04dk.pt" % (bert_clf_steps // 1000))
            torch.save(model, ckpt)
            logger.info("Bert classifier saved at %s.", ckpt)
            model.to(device)
            model.train()

        if global_step >= bert_clf_steps:
            break
    model.eval()
    return model


class BertClfPrediction(MeasurementBase):
    """Generate BERT prediction for paraphrase."""

    def __init__(self, dataset_name, trainset, testset, bert_gpu_id=-1,
                 bert_clf_steps=20000, bert_clf_bs=32, bert_clf_lr=0.00002,
                 bert_clf_optimizer="adamw", bert_clf_weight_decay=0.001,
                 bert_clf_period_summary=100, bert_clf_period_val=500,
                 bert_clf_period_save=5000, bert_clf_val_steps=10, **kargs):
        super(BertClfPrediction, self).__init__()

        if trainset["cased"]:
            model_init = "bert-base-cased"
            logger.info(
                "Use cased model in Bert classifier prediction flip measurement.")
        else:
            model_init = "bert-base-uncased"
            logger.info(
                "Use uncased model in Bert classifier prediction flip measurement.")

        self._tokenizer = BertTokenizer.from_pretrained(model_init)
        if bert_gpu_id == -1:
            logger.warning("Bert measurement is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("Bert measurement is running on GPU %d.", bert_gpu_id)
            self._device = torch.device("cuda:%d" % bert_gpu_id)

        self._model = load_or_train_bert_clf(
            tokenizer=self._tokenizer,
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
        if text1 is None:
            seq = ["[CLS]"] + self._tokenizer.tokenize(text0)
            seq_tensor = torch.tensor(self._tokenizer.convert_tokens_to_ids(seq)).to(self._device)
            seq_tensor = seq_tensor[:200]
            return self._model(seq_tensor.unsqueeze(0))[0][0].detach().cpu().numpy()
        else:
            seq0 = self._tokenizer.tokenize(text0)
            seq1 = self._tokenizer.tokenize(text1)
            l0 = len(seq0)
            l1 = len(seq1)
            seq_tensor = torch.tensor(
                self._tokenizer.convert_tokens_to_ids(["[CLS]"] + seq0 + seq1)
            ).to(self._device).unsqueeze(0)
            tok_type = torch.tensor(
                [0] * (l0 + 1) + [1] * l1).to(self._device).unsqueeze(0)
            seq_tensor = seq_tensor[:, :200]
            tok_type = tok_type[:, :200]
            return self._model(seq_tensor, token_type_ids=tok_type)[0][0].detach().cpu().numpy()

    def predict(self, text0, text1):
        return int(np.argmax(self.predict_raw(text0, text1)))

    def __call__(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        if paraphrase_field == "text0":
            return self.predict(paraphrase, None)
        else:
            assert paraphrase_field == "text1"
            return self.predict(data_record["text0"], paraphrase)
