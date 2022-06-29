import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer

from fibber import resources


class BertClassifierDARTS(nn.Module):
    def __init__(self,
                 model_type,
                 output_dim=2,
                 freeze_bert=True,
                 ensemble=0,
                 N=5,
                 inference=False,
                 is_training=True,
                 temperature=1.0,
                 gumbel=0,
                 scaler=0,
                 darts=True,
                 device='cpu'):

        super(BertClassifierDARTS, self).__init__()
        self.name = model_type
        self.scaler = scaler
        self.gumbel = gumbel
        self.embedding_matrix = []
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(resources.get_transformers(model_type))
        self.bert_layer = AutoModel.from_pretrained(resources.get_transformers(model_type))

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.feature_dim = 768
        self.output_dim = output_dim
        self.ensemble = ensemble
        self.N = N
        self.inference = inference
        self.flg_training = is_training
        self.emb1 = None
        self.temperature = temperature
        self.embedding = self.bert_layer.get_input_embeddings()
        self.dropout = nn.Dropout(0.5)
        self.L = 3
        self.sample_wise_training = True
        self.darts = darts

        def forward_hook(module, input, output):
            self.emb1 = output

        for module in self.bert_layer.modules():
            if isinstance(module, nn.Embedding):
                if module.weight.size()[0] >= 30000:
                    self.embedding_weight = module.weight.data.clone()
                    self.embedding_dim = self.embedding_weight.size()[1]

        if self.ensemble:
            if self.darts:
                self.heads = nn.ModuleList([
                    nn.ModuleList([
                        nn.Sequential(nn.Linear(self.feature_dim, self.output_dim)),
                        nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
                                      nn.Linear(self.feature_dim, self.output_dim)),
                        nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
                                      nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
                                      nn.Linear(self.feature_dim, self.output_dim)),
                    ]) for i in range(self.N)
                ])

                self.darts_decision = nn.ParameterList([
                    torch.nn.Parameter(torch.randn(self.L)) for i in range(self.N)
                ])

            else:
                self.heads = nn.ModuleList(
                    [nn.ModuleList([nn.Sequential(nn.Linear(self.feature_dim, self.output_dim))])
                     for i in range(self.N)])

            self.switch = nn.Linear(self.N * self.output_dim + self.feature_dim, self.N)
            self.probs = torch.ones(self.N).to(self.device).float()

        else:
            self.feature2main = nn.Linear(self.feature_dim, self.output_dim)

        self.to(self.device)

    def select_head_gumbel_single(self, base_pred, pred_heads):
        pred_heads = torch.cat((pred_heads, base_pred), 1)
        gumbel_t = self.switch(pred_heads).sum(0)  # [N]

        if self.gumbel > 0:
            self.probs = F.softmax(self.add_gumbel(gumbel_t) * self.temperature, dim=-1)
            gumbel_t = gumbel_t * self.probs

        return gumbel_t

    def select_head_gumbel(self, base_pred, pred_heads):
        pred_heads = torch.cat(pred_heads, 1)  # batch size * (N * num_class)
        all_gumbel_t = []

        for j in range(len(pred_heads)):
            gumbel_t_single = self.select_head_gumbel_single(base_pred[j:j + 1],
                                                             pred_heads[j:j + 1])
            all_gumbel_t.append(gumbel_t_single.unsqueeze(0))
        gumbel_t = torch.cat(all_gumbel_t, 0)

        return gumbel_t

    def forward_from_embedding(self, emb, attn_masks):
        cont_reps = self.bert_layer(inputs_embeds=emb, attention_mask=attn_masks)
        pred = cont_reps.last_hidden_state[:, 0]

        pred = self.dropout(pred) if not self.inference else pred
        pred = pred.view(-1, self.feature_dim)

        if self.ensemble > 0:
            self.pred_heads = []
            for i in range(len(self.heads)):
                layer = self.heads[i]
                if self.darts:
                    if not self.inference:
                        controls = F.softmax(self.darts_decision[i], dim=-1)
                        scores = [(layer[j](pred) * controls[j]).unsqueeze(0) for j in
                                  range(len(layer))]
                        scores = torch.mean(torch.cat(scores, 0), 0)

                    else:
                        controls = F.softmax(self.darts_decision[i], dim=-1)
                        idx = torch.argmax(controls)
                        scores = layer[idx](pred)

                    scores = scores.view(-1, self.output_dim)
                    self.pred_heads.append(scores)

                else:
                    self.pred_heads.append(layer[0](pred))

            self.random_key = self.select_head_gumbel(pred, self.pred_heads)
            pred = torch.cat(
                [(self.pred_heads[i] * self.random_key[:, i].unsqueeze(1)).unsqueeze(1) for i in
                 range(len(self.pred_heads))], 1)
            pred = torch.mean(pred, 1)

        else:
            key1 = self.feature2main(pred)
            pred = key1

        return pred

    def forward(self, input_ids, attention_mask, **kwargs):
        self.emb1 = Variable(self.embedding(input_ids), requires_grad=True)
        return [self.forward_from_embedding(self.emb1, attention_mask)]

    def add_gumbel(self, o_t, eps=1e-10):
        u = torch.zeros(o_t.size())
        u = u.to(self.device)
        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t

        return gumbel_t

    def init_linear(self):
        for name, param in self.named_parameters():
            if 'heads.' in name and param.requires_grad and len(param.shape) > 0:
                1 / math.sqrt(param.shape[0])
