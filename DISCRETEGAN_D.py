# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_D.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import torch
import torch.nn as nn
import torch.optim as optim
from models.discriminator import CNNDiscriminator

dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]


class DISCRETEGAN_D(CNNDiscriminator):
    def __init__(self, embed_dim, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(DISCRETEGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx, gpu,
                                       dropout)
        self.criterion = nn.CrossEntropyLoss() # ce = logsoftmax + nll
        self.opt = optim.Adam(self.parameters(), lr=1e-2)

    def get_gradient_estimation(self, samples):
        '''
        search nearest neighbors as probability estimation
        samples: b x s
        ===
        return:
        (-)gradient estimation b x s x v
        '''
        pred = self.forward(samples) # bx2
        emb_mat = self.embeddings.weight.clone().detach() # v * embedding_dim
        target = torch.ones(pred.size(0)).long()
        if self.gpu:
            target = target.cuda()
        loss = self.criterion(pred, target)
        # update embedding
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # select the embeddings of input tokens
        emb = self.embeddings(samples) # b x s x embedding_dim
        # find the nearest embedding in original space by cos similarity
        # if same, then all element would be 1, which means no change

        emb_mat_norm = torch.norm(emb_mat, dim=-1) # v,
        emb_norm = torch.norm(emb, dim=-1) # b*s,
        sim_norm = torch.matmul(emb_norm.unsqueeze(-1), emb_mat_norm.unsqueeze(0))
        emb_sim = torch.matmul(emb, emb_mat.transpose(0,1)) / sim_norm # b*s*v
        estimated_gradient = emb_sim - torch.mean(emb_sim, dim=-1).unsqueeze(-1) # b*s*v

        return estimated_gradient.detach()