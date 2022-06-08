import numpy as np
import torch
import torch.nn.functional as F

from fibber.datasets.dataset_utils import text_md5
from fibber.defense_strategies.defense_strategy_base import DefenseStrategyBase
from fibber.metrics.bert_lm_utils import get_lm


def lmag_fix_sentences(sentences, context, tokenizer, lm, clf, device, bs=50, rep=10):
    assert bs % rep == 0

    st = 0
    ret = []

    sentences_t = sentences
    sentences = []
    for item in sentences_t:
        sentences += [item] * rep

    if context is not None:
        context_t = context
        context = []
        for item in context_t:
            context += [item] * rep

    while st < len(sentences):
        ed = min(st + bs, len(sentences))
        if context is not None:
            batch_input_small = tokenizer(context_t[st // rep:ed // rep],
                                          sentences_t[st // rep:ed // rep],
                                          padding=True)
            batch_input = tokenizer(context[st:ed], sentences[st:ed], padding=True)
        else:
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
                rng = np.random.RandomState(
                    hash(int(text_md5(sentences_t[i // rep]), 16) % 1000000007))
                if context is not None:
                    change = -1
                    for j in range(1, token_type_ids.size(1)):
                        if token_type_ids[i - st, j - 1] == 0 and token_type_ids[i - st, j] == 1:
                            change = j
                            break
                    u = change
                    v = attention_mask[i - st].sum() - 1
                else:
                    u = 1
                    v = attention_mask[i - st].sum() - 1

                n_mask = max(1, int((v - u) * 0.2))
                prob = attention_grad[(i - st) // rep, u:v]
                alpha = 0.6
                prob = prob ** float(alpha)
                prob /= np.sum(prob)

            samples = u + rng.choice(v - u, p=prob, replace=False, size=n_mask)
            for p in samples:
                input_ids[i - st, p] = tokenizer.mask_token_id
        pred = lm(
            input_ids=input_ids.to(device),
            token_type_ids=token_type_ids.to(device),
            attention_mask=attention_mask.to(device)
        )[0].argmax(dim=-1).detach().cpu().numpy()

        for i in range(st, ed):
            if context is not None:
                change = -1
                for j in range(1, token_type_ids.size(1)):
                    if token_type_ids[i - st, j - 1] == 0 and token_type_ids[i - st, j] == 1:
                        change = j
                        break
                u = change
                v = attention_mask[i - st].sum() - 1
            else:
                u = 1
                v = attention_mask[i - st].sum() - 1

            ret.append(tokenizer.decode(pred[i - st, u:v], skip_special_tokens=True))

        st = ed
    return ret


class LMAgStrategy(DefenseStrategyBase):
    """Base class for Tuning strategy"""

    def input_manipulation(self):
        pass

    def fit(self, trainset, save_path):
        _, self._lm = get_lm("finetune", self._dataset_name, trainset, self._device)
        self._lm = self._lm.eval().to(self._device)
        self._lmag_repeat = 10
