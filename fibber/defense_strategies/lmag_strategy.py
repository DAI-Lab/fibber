from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from fibber.defense_strategies.defense_strategy_base import DefenseStrategyBase
from fibber.metrics.bert_lm_utils import get_lm
from fibber.metrics.classifier.input_manipulation_classifier import InputManipulationClassifier


def lmag_fix_sentences(sentences, data_record_list, tokenizer, lm, clf, device, bs=50, rep=10):
    assert bs % rep == 0

    st = 0
    ret = []

    sentences_t = sentences
    sentences = []
    for item in sentences_t:
        sentences += [item] * rep

    while st < len(sentences):
        ed = min(st + bs, len(sentences))
        batch_input_small = tokenizer(sentences_t[st // rep:ed // rep], padding=True)
        batch_input = tokenizer(sentences[st:ed], padding=True)

        input_ids = torch.tensor(batch_input_small["input_ids"])
        token_type_ids = torch.tensor(batch_input_small["token_type_ids"])
        attention_mask = torch.tensor(batch_input_small["attention_mask"])

        embeddings = clf.bert.embeddings(
            input_ids=input_ids.to(device),
            token_type_ids=token_type_ids.to(device)
        )

        embeddings = embeddings.detach().clone()
        embeddings.requires_grad = True

        max_prob = torch.sum(torch.max(
            F.log_softmax(clf(
                attention_mask=attention_mask.to(device),
                inputs_embeds=embeddings
            )[0], dim=1), dim=1)[0])
        max_prob.backward()

        embeddings_grad = embeddings.grad.detach()
        attention_grad = torch.sum(embeddings_grad * embeddings_grad, dim=-1).cpu().numpy()
        attention_grad = np.sqrt(attention_grad)
        del embeddings
        del max_prob
        del embeddings_grad
        torch.cuda.empty_cache()

        input_ids = torch.tensor(batch_input["input_ids"])
        token_type_ids = torch.tensor(batch_input["token_type_ids"])
        attention_mask = torch.tensor(batch_input["attention_mask"])

        for i in range(st, ed):
            if i % rep == 0:
                u = 1
                v = attention_mask[i - st].sum() - 1

                n_mask = max(1, int((v - u) * 0.2))
                prob = attention_grad[(i - st) // rep, u:v]
                alpha = 0.6
                prob = prob ** float(alpha)
                prob /= np.sum(prob)

            samples = u + np.random.choice(v - u, p=prob, replace=False, size=n_mask)
            for p in samples:
                input_ids[i - st, p] = tokenizer.mask_token_id
        pred = lm(
            input_ids=input_ids.to(device),
            token_type_ids=token_type_ids.to(device),
            attention_mask=attention_mask.to(device)
        )[0].argmax(dim=-1).detach().cpu().numpy()

        for i in range(st, ed):
            u = 1
            v = attention_mask[i - st].sum() - 1

            ret.append(tokenizer.decode(pred[i - st, u:v], skip_special_tokens=True))

        st = ed

    ret_reformat = []
    for i in range(0, len(sentences), rep):
        ret_reformat.append(ret[i:i + rep])
    return ret_reformat


class LMAgStrategy(DefenseStrategyBase):
    """Base class for Tuning strategy"""

    __abbr__ = "lmag"
    __hyperparameters__ = [
        ("reps", int, 10, "number of rewrites.")]

    def fit(self, trainset):
        self._tokenizer, self._lm = get_lm("finetune", self._dataset_name, trainset, self._device)
        self._lm = self._lm.eval().to(self._device)
        self._lmag_repeat = self._strategy_config["reps"]

    def load(self, trainset):
        self.fit(trainset)
        return InputManipulationClassifier(
            self._classifier,
            partial(lmag_fix_sentences, tokenizer=self._tokenizer, lm=self._lm,
                    clf=self._classifier._model, device=self._classifier._device,
                    rep=self._lmag_repeat),
            str(self._classifier),
            field=self._classifier._field,
            bs=self._classifier._bs)
