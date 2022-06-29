import numpy as np
import torch
import tqdm
from transformers import (
    BertForSequenceClassification, DistilBertForSequenceClassification,
    RobertaForSequenceClassification)


class EmbeddingGradHook:
    def __init__(self):
        self.output = None
        self._enable = True

    def disable(self):
        self._enable = False
        self.output = None

    def enable(self):
        self._enable = True

    def __call__(self, module, input_, output_):
        if self._enable:
            self.output = output_
            output_.retain_grad()
        else:
            self.output = None


def pick_diag(x):
    ret = []
    for i in range(x.size(0)):
        ret.append(x[i, i])
    return torch.stack(ret, dim=0)


def estimate_weight(clf_model, embedding_layer, vocabulary, hook, example, token_ids_raw,
                    anchor_word_tokenizer_id, batch_size, device):
    hook.enable()
    sent_len = len(token_ids_raw)
    batch_input_raw = torch.tensor([token_ids_raw]).to(device)

    if anchor_word_tokenizer_id is not None:
        input_text = [token_ids_raw[:i] + [anchor_word_tokenizer_id] + token_ids_raw[i + 1:]
                      for i in range(sent_len)]
        log_prob = []
        grad = []
        emb_output = []

        for st in range(0, len(input_text), batch_size):
            ed = min(st + batch_size, len(input_text))
            batch_input = torch.tensor(input_text[st:ed]).to(device)
            output = clf_model(batch_input, return_dict=True)
            log_prob_tmp = torch.log_softmax(output["logits"], dim=1)[:, example["label"]]
            log_prob.append(log_prob_tmp.clone())
            torch.sum(log_prob_tmp).backward()
            grad.append(hook.output.grad.clone())
            emb_output.append(hook.output.clone())
            del output
        log_prob = torch.cat(log_prob, dim=0)
        grad = pick_diag(torch.cat(grad, dim=0))
        emb_output = pick_diag(torch.cat(emb_output, dim=0))
    else:
        output = clf_model(batch_input_raw, return_dict=True)
        log_prob = torch.log_softmax(output["logits"], dim=1)[0, example["label"]]
        log_prob.backward()
        log_prob = log_prob.detach()
        grad = hook.output.grad.clone()[0]  # L * d
        emb_output = hook.output.clone()[0]  # L * d
        del output

    hook.disable()

    with torch.no_grad():
        weight = []
        for st in range(0, len(vocabulary), batch_size):
            ed = min(st + batch_size, len(vocabulary))
            # get a batch of words
            tmp = np.asarray([item[1] for item in vocabulary[st:ed]])
            # repeat each word L times
            tmp = np.tile(tmp.reshape(-1, 1), (1, grad.size(0)))  # batch * L

            weight.append(
                (torch.einsum("bld,ld->bl", embedding_layer(torch.tensor(tmp).to(device)), grad)
                 - torch.einsum("ld,ld->l", emb_output, grad)) + log_prob)

        weight = torch.cat(weight, dim=0)  # V * l
        weight[:, 0] = 1e8   # change clf tok weight to inf
        weight[:, -1] = 1e8   # change eos tok to inf
    return weight.detach()


def solve_euba(clf_model, tokenizer, vocabulary, val_set,
               use_mask, use_top1, early_stop, batch_size, device):
    incorrect = np.zeros(len(vocabulary))
    detail_incorrect = np.zeros((len(vocabulary), len(val_set["label_mapping"])))

    early_stop_batch_cnt = max(early_stop // batch_size, 1)

    if isinstance(clf_model, DistilBertForSequenceClassification):
        lm_model = clf_model.distilbert
    elif isinstance(clf_model, BertForSequenceClassification):
        lm_model = clf_model.bert
    elif isinstance(clf_model, RobertaForSequenceClassification):
        lm_model = clf_model.roberta
    else:
        raise RuntimeError("unknown classifier.")

    embedding_layer = lm_model.embeddings.word_embeddings
    hook = EmbeddingGradHook()
    embedding_layer.register_forward_hook(hook)

    iter_idx = 0
    pbar = tqdm.tqdm(val_set["data"])
    for example in pbar:
        iter_idx += 1
        token_ids_raw = tokenizer(example["text0"])["input_ids"]
        sent_len = len(token_ids_raw)
        batch_input_raw = torch.tensor([token_ids_raw]).to(device)

        if use_mask:
            weight = estimate_weight(
                clf_model, embedding_layer, vocabulary, hook, example, token_ids_raw,
                anchor_word_tokenizer_id=tokenizer.mask_token_id, batch_size=batch_size,
                device=device)
        else:
            weight = estimate_weight(
                clf_model, embedding_layer, vocabulary, hook, example, token_ids_raw,
                anchor_word_tokenizer_id=None, batch_size=batch_size, device=device)

        with torch.no_grad():
            sorted_index = torch.argsort(weight.reshape(-1)).detach().cpu().numpy()
            weight = weight.detach().cpu().numpy()

            word_verify_cnt = [0] * len(vocabulary)
            # if >= 0, the number of positions tried. if -1, attack succeed.

            # weight-based trail
            miss_counter = 0
            current = 0
            while current < len(sorted_index):
                input_ids = batch_input_raw.repeat(batch_size, 1)
                batch_info = []
                jj = 0

                while current < len(sorted_index) and jj < batch_size:
                    wid = sorted_index[current] // sent_len
                    pos = sorted_index[current] % sent_len
                    current += 1

                    if pos == 0 or pos == sent_len - 1:
                        continue

                    if word_verify_cnt[wid] == -1:
                        continue

                    if use_top1 and word_verify_cnt[wid] > 0:
                        continue

                    input_ids[jj, pos] = vocabulary[wid][1]
                    word_verify_cnt[wid] += 1
                    batch_info.append(wid)
                    jj += 1

                preds = torch.argmax(clf_model(input_ids)["logits"], dim=1).detach().cpu().numpy()

                hit = False
                for j in range(len(batch_info)):
                    if preds[j] != example["label"]:
                        if word_verify_cnt[batch_info[j]] != -1:
                            word_verify_cnt[batch_info[j]] = -1
                            incorrect[batch_info[j]] += 1
                            detail_incorrect[batch_info[j], example["label"]] += 1
                            hit = True

                if not hit:
                    miss_counter += 1
                    if miss_counter == early_stop_batch_cnt:
                        break
                else:
                    miss_counter = 0

            # top words trail
            sorted_index = np.argsort(-detail_incorrect[:, example["label"]])
            current = 0
            miss_counter = 0
            while current < len(sorted_index):
                input_ids = batch_input_raw.repeat(batch_size, 1)
                batch_info = []
                jj = 0

                while current < len(sorted_index) and jj < batch_size:
                    wid = sorted_index[current]
                    pos = np.argmin(weight[wid])
                    current += 1
                    if pos == 0 or pos == sent_len - 1:
                        continue
                    if word_verify_cnt[wid] != 0:
                        continue
                    input_ids[jj, pos] = vocabulary[wid][1]
                    word_verify_cnt[wid] += 1
                    batch_info.append(wid)
                    jj += 1

                preds = torch.argmax(clf_model(input_ids)["logits"], dim=1).detach().cpu().numpy()
                hit = False
                for j in range(len(batch_info)):
                    if preds[j] != example["label"]:
                        if word_verify_cnt[batch_info[j]] != -1:
                            word_verify_cnt[batch_info[j]] = -1
                            incorrect[batch_info[j]] += 1
                            detail_incorrect[batch_info[j], example["label"]] += 1
                            hit = True
                if not hit:
                    miss_counter += 1
                    if miss_counter == early_stop_batch_cnt:
                        break
                else:
                    miss_counter = 0

        if iter_idx % 10 == 0:
            result = [(w[0], eps) for w, eps in zip(vocabulary, incorrect)]
            result = sorted(result, key=lambda x: -x[1])
            print(result[:10])

    return incorrect / len(val_set["data"]), detail_incorrect
