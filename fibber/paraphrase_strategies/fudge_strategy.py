import os.path

import numpy as np
import torch
import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from fibber import resources
from fibber.metrics.classifier.transformer_classifier import get_optimizer
from fibber.paraphrase_strategies.strategy_base import StrategyBase


def make_input_output_pair(tokenizer, x):
    """Tokenize the text, then construct input and output for GPT2."""
    toks = tokenizer.encode(x, add_special_tokens=True)

    (len(toks)) // 2
    output = toks[:]
    # output[:half] = -100
    return toks, output


def make_batch(toks_list):
    """Convert multiple text to a batch tensor."""
    n = len(toks_list)
    max_len = max([len(x) for x in toks_list])

    ids = np.zeros((n, max_len), dtype='int')
    mask = np.zeros((n, max_len), dtype='int')

    for i, item in enumerate(toks_list):
        ids[i, :len(item)] = np.asarray(item)
        mask[i, :len(item)] = 1

    return ids, mask


class FudgeStrategy(StrategyBase):
    """A baseline paraphrase strategy. Just return the reference."""

    __abbr__ = "fu"

    def fit(self, trainset):
        self._sim_metric = self._metric_bundle.get_metric("USESimilarityMetric")
        self._clf_metric = self._metric_bundle.get_target_classifier()

        gpt2_pretrained_model = "gpt2-medium"
        self._tokenizer = GPT2TokenizerFast.from_pretrained(
            resources.get_transformers(gpt2_pretrained_model))

        path = "gpt2-%s-1" % self._dataset_name

        if os.path.exists(path):
            self._model = GPT2LMHeadModel.from_pretrained(path).to(self._device)
            return

        self._model = GPT2LMHeadModel.from_pretrained(
            resources.get_transformers(gpt2_pretrained_model)).to(
            self._device)

        opt, sche = get_optimizer("adam", lr=0.0001, decay=0.001,
                                  params=self._model.parameters(), train_step=5000)

        self._model.train()
        for i in tqdm.tqdm(range(5000)):
            batch = np.random.choice(trainset["data"], size=8)
            text = [item["text0"] for item in batch]
            input_output = [
                make_input_output_pair(
                    self._tokenizer, "<|endoftext|> " + x + " <|endoftext|> " + x) for x in text]
            input, output = zip(*input_output)

            toks_input, mask = make_batch(input)
            toks_output, _ = make_batch(output)
            toks_output = toks_output * mask - 100 * (1 - mask)
            toks_input = torch.tensor(toks_input)
            toks_output = torch.tensor(toks_output)
            mask = torch.tensor(mask)

            output = self._model(toks_input.to(self._device), attention_mask=mask.to(self._device),
                                 labels=toks_output.to(self._device))
            opt.zero_grad()
            output[0].backward()
            opt.step()
            sche.step()
            if i % 100 == 0:
                print(output[0])
        self._model.eval()
        self._model.save_pretrained(path)

    def score(self, data_record, tmp, text, ll):
        t0_for_use = self._tokenizer.decode(tmp[1:ll + 1])
        sim = self._sim_metric.measure_batch(t0_for_use, text)

        dist = self._clf_metric.predict_log_dist_batch(t0_for_use, text)
        label = data_record["label"]

        # print(t0_for_use, text)
        correct_prob = (dist[:, label]).copy()
        dist[:, label] = -1e8
        incorrect_prob = np.max(dist, axis=1)

        return -np.maximum(correct_prob - incorrect_prob, 0) - 10 * \
            np.maximum(0.95 - np.asarray(sim), 0) ** 2

    def paraphrase_example(self, data_record, n):
        tmp, _ = make_input_output_pair(
            self._tokenizer,
            "<|endoftext|> " + data_record["text0"] + " <|endoftext|> " + data_record["text0"])
        batch = torch.tensor([tmp]).to(self._device)

        topk = 20
        ret = []
        for _ in range(n):
            for i in range(len(tmp) // 2 + 2, len(tmp) - 1):
                lm_logits = self._model(batch[:, :i + 1])[0][0, -1, :]
                lm_logits[self._tokenizer.eos_token_id] = -1e8
                lm_prob = torch.log_softmax(lm_logits, dim=0)
                kth_vals, kth_idx = lm_prob.topk(topk, dim=-1)

                kth_idx = kth_idx.detach().cpu().numpy()
                batch_np = batch.detach().cpu().numpy()

                text = [list(batch_np[0][len(tmp) // 2 + 2:i + 1]) + [t] for t in kth_idx]
                text = [self._tokenizer.decode(t) for t in text]
                score = self.score(data_record, tmp, text, i - len(tmp) // 2 - 1)

                kth_vals += torch.tensor(score).to(self._device)
                kth_vals = torch.softmax(kth_vals, dim=0).detach().cpu().numpy()

                pick = np.random.choice(kth_idx, p=kth_vals)
                batch[0][i + 1] = pick

            ret.append(
                self._tokenizer.decode(
                    batch[0].detach().cpu().numpy()[len(tmp) // 2 + 2:len(tmp) - 1]))
        return ret
