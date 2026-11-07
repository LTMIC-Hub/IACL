#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal

class IACL(nn.Module):
    '''
    Implicit augmentation contrastive loss
    '''
    def __init__(self, feature_num=None, cls_num_list=None, temperature=10,):  
        super(IACL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list
        self.class_num = len(self.cls_num_list)
        self.feature_num = feature_num
        self.num = torch.zeros(self.class_num).cuda() 
        self.weight = torch.zeros(self.class_num).cuda()
        self.Variance = torch.zeros(self.class_num, self.feature_num, self.feature_num).cuda()


    def pairwise_cosine_sim(self, x, y):
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)
        return torch.matmul(x, y.T)


    def upper_bound_contrastive_loss(self, feat, labels, cov, ratio, temperature):
        device = feat.device
        N, D = feat.shape
        batch_size = N


        mask = torch.eq(labels, labels.T).float().cuda()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask


        sim_mat = torch.matmul(feat, feat.T)
        term_1 = (sim_mat.unsqueeze(1) - sim_mat.unsqueeze(2)) 
        #term_1 = (sim_mat.unsqueeze(2) - sim_mat.unsqueeze(1))

        sigma_feat = torch.matmul(cov.unsqueeze(1), feat.unsqueeze(0).unsqueeze(-1)).squeeze(-1)  

        diag_term = (sigma_feat * feat.unsqueeze(0)).sum(-1)                                 
        cross_term = torch.matmul(sigma_feat, feat.T).transpose(1, 2)                        

        diag_j = diag_term.unsqueeze(2)         
        diag_k = diag_term.unsqueeze(1)           

        term_2 =  (diag_j + diag_k - 2.0 * cross_term)                                         
        logits = (term_1 / temperature + ratio * term_2 / (2.0 * (temperature ** 2)))

        k_mask = ~torch.eye(N, dtype=torch.bool, device=device)
        logits = logits.masked_fill(~k_mask[:, None, :], -1e30)

        log_term = torch.logsumexp(logits, dim=-1) 
        loss = (log_term * mask).sum(dim=1) / mask.sum(dim=1)
        loss = loss.mean()

        return loss
    

    def Covariance_matrix_calibration(self, features, targets, Variance, total_cov, out_cov, threshold):
        unique = torch.unique(targets)
       
        targets = targets.contiguous().view(-1, 1)
        original_target = targets.repeat(2, 1).squeeze()
        original_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        for index_1, i in enumerate(unique):

            if i >= threshold[0]:
                index = torch.where(original_target == i)
                temp_cov = out_cov[index]
                temp_Variance = torch.sum(temp_cov, dim=0) / index[0].shape[0]
                self.num[i] = self.num[i] + index[0].shape[0]
                self.weight[i] = index[0].shape[0] / self.num[i]
                self.Variance[i] = (1 - self.weight[i]) * self.Variance[i] + self.weight[i] * temp_Variance
                total_cov[index[0]] = out_cov[index[0]]  + self.Variance[i]            

            if i >= threshold[1]:
                index = torch.where(original_target == i)
                tail_features = original_features[index[0]]
                cos_dis = self.pairwise_cosine_sim(tail_features, original_features)
                ones = torch.ones(original_features.shape[0]).cuda()
                ones[index[0]] = 0
                mask = cos_dis * ones  
                a, sort_idx = torch.sort(mask, dim=1, descending=True)
                top_k_idx = sort_idx[:,:1].view(-1)
                top_k_idx = top_k_idx.squeeze()
                temp_Variance = out_cov[top_k_idx]
                total_cov[index[0]] = total_cov[index[0]] + temp_Variance

        return total_cov

    def forward(self, features, targets, epoch, out_cov, threshold):
        '''
        :param proxy: proxy vector
        :param features: feature vector
        :param targets: labels
        '''
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)

        if epoch > 40:
            total_out_cov = self.Covariance_matrix_calibration(features, targets, self.Variance, out_cov, out_cov.detach(), threshold)  
        else:
            total_out_cov = out_cov

        targets = torch.cat([targets.repeat(2, 1)], dim=0).cuda()
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
       
        ratio = ((epoch+1) / 100)
        total_out_cov = total_out_cov.squeeze()
        loss = self.upper_bound_contrastive_loss(features, targets, total_out_cov, ratio, self.temperature)
        return loss



class BCL(nn.Module):
    '''
    Balanced Contrastive loss
    '''
    def __init__(self,cls_num_list=None, temperature=0.1,):
        super(BCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list


    def forward(self, features, targets):
        '''
        :param features: feature vector
        :param targets: labels
        '''
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)

        targets = torch.cat([targets.repeat(2, 1)], dim=0).cuda()
        batch_cls_count = torch.eye(len(self.cls_num_list)).cuda()[targets].sum(dim=0).squeeze()
        
        mask = torch.eq(targets, targets.T).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # get similarity matrix
        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        logits = features.mm(features.T)
        logits = torch.div(logits, self.temperature)
    
      
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
       
        # class averaging
        exp_logits = torch.exp(logits) * logits_mask
        
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
    
        # get loss
        log_prob = logits - torch.log(exp_logits_sum)
     
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
        loss = - mean_log_prob_pos
        loss = loss.mean()
      
        return loss

