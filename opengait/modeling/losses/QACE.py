import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import BaseLoss

class QACE(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps_smooth=0.1, loss_term_weight=1.0, log_accuracy=False,
                 loss_tpe=None, angular_margin=None, add_margin=None, h=0.333, log_margin=False,
                 angular_margin_all=0.0, add_margin_all=0.0):
        super(QACE, self).__init__(loss_term_weight)

        self.label_smooth = label_smooth
        self.eps_smooth = eps_smooth
        self.log_accuracy = log_accuracy
        # self.scale = scale
        self.loss_tpe = loss_tpe
        
        if loss_tpe == 'ArcFace':
            self.angular_margin = angular_margin
            self.eps_adaFace = 1e-4
            self.scale = scale
            self.angular_margin_all = angular_margin_all
            self.add_margin_all = add_margin_all
        elif loss_tpe == 'CosFace':
            self.add_margin = add_margin
            self.eps_adaFace = 1e-4
            self.scale = scale
            self.angular_margin_all = angular_margin_all
            self.add_margin_all = add_margin_all
        elif loss_tpe == 'AdaFace':
            self.m = angular_margin
            self.eps_adaFace = 1e-4
            self.scale = scale
            self.h = h
            self.log_margin = log_margin
            self.angular_margin_all = angular_margin_all
            self.add_margin_all = add_margin_all
        else:
            self.scale = scale
            pass
            # raise NotImplementedError

    def forward(self, feature_hpp, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()

        if self.loss_tpe == 'ArcFace':
            # ArcFace
            cosine = logits.view(n, c*p).clamp(-1+self.eps_adaFace, 1-self.eps_adaFace)  # for stable
            theta = torch.acos(cosine)

            m_hot_angular = torch.zeros(n, c, device=labels.device).scatter_(
                1, labels.unsqueeze(-1), self.angular_margin).unsqueeze(2).repeat(1, 1, p)  # [n, c, p]
            m_hot_angular = m_hot_angular.view(n, c*p)

            theta_m = torch.clip(theta + m_hot_angular + self.angular_margin_all, min=self.eps_adaFace, max=math.pi-self.eps_adaFace)
            cosine_m = theta_m.cos()
            logits = cosine_m.view(n, c, p)

            log_preds = F.log_softmax(logits * self.scale, dim=1)  # [n, c, p]
            one_hot_labels = self.label2one_hot(
                labels, c).unsqueeze(2).repeat(1, 1, p)  # [n, c, p]
            loss = self.compute_loss(log_preds, one_hot_labels)
            self.info.update({'ArcFaceloss': loss.detach().clone()})
            if self.log_accuracy:
                pred = logits.argmax(dim=1)  # [n, p]
                accu = (pred == labels.unsqueeze(1)).float().mean()
                self.info.update({'accuracy': accu})

        elif self.loss_tpe == 'CosFace':
            # CosFace
            cosine = logits.view(n, c*p).clamp(-1+self.eps_adaFace, 1-self.eps_adaFace)  # for stable

            m_hot_add = torch.zeros(n, c, device=labels.device).scatter_(
                1, labels.unsqueeze(-1), self.add_margin).unsqueeze(2).repeat(1, 1, p)  # [n, c, p]
            m_hot_add = m_hot_add.view(n, c*p)

            cosine_m = cosine - m_hot_add + self.add_margin_all
            logits = cosine_m.view(n, c, p)

            log_preds = F.log_softmax(logits * self.scale, dim=1)  # [n, c, p]
            one_hot_labels = self.label2one_hot(
                labels, c).unsqueeze(2).repeat(1, 1, p)  # [n, c, p]
            loss = self.compute_loss(log_preds, one_hot_labels)
            self.info.update({'CosFaceloss': loss.detach().clone()})
            if self.log_accuracy:
                pred = logits.argmax(dim=1)  # [n, p]
                accu = (pred == labels.unsqueeze(1)).float().mean()
                self.info.update({'accuracy': accu})

        elif self.loss_tpe == 'AdaFace':
            
            # Feature Norm
            feature_norm = torch.norm(feature_hpp, p=2, dim=1, keepdim=True)  # [n, 1, p]
            feature_norm = torch.mean(feature_norm, dim=-1, keepdim=False)     # [n, 1, 1]
            safe_feature_norms = torch.clip(feature_norm, min=0.001, max=100) # for stability   # [n, 1, 1]
            safe_feature_norms = safe_feature_norms.clone().detach()          # [n, 1, 1]

            with torch.no_grad():
                mean = safe_feature_norms.mean().detach()  # [1]
                std = safe_feature_norms.std().detach()  # [1]
            
            margin_scaler = (safe_feature_norms - mean) / (std + self.eps_adaFace) # 66% between -1, 1
            margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
            margin_scaler = torch.clip(margin_scaler, -1, 1)  # [n, 1, 1]
            g_angular = self.m * margin_scaler * (-1)
            g_add = self.m + (self.m * margin_scaler)

            cosine = logits.view(n, c*p).clamp(-1+self.eps_adaFace, 1-self.eps_adaFace)  # for stable
            theta = torch.acos(cosine)

            ## Angular
            m_hot_angular = torch.zeros(n, c, device=labels.device).scatter_(
                1, labels.unsqueeze(-1), 1).unsqueeze(2).repeat(1, 1, p)  # [n, c, p]
            m_hot_angular = m_hot_angular.view(n, c*p)
            m_hot_angular = m_hot_angular * g_angular
            # ## Addition
            m_hot_add = torch.zeros(n, c, device=labels.device).scatter_(
                1, labels.unsqueeze(-1), 1).unsqueeze(2).repeat(1, 1, p)  # [n, c, p]
            m_hot_add = m_hot_add.view(n, c*p)
            m_hot_add = m_hot_add * g_add

            theta_m = torch.clip(theta + m_hot_angular + self.angular_margin_all, min=self.eps_adaFace, max=math.pi-self.eps_adaFace)
            cosine_m = theta_m.cos() - m_hot_add + self.add_margin_all
            logits = cosine_m.view(n, c, p)

            log_preds = F.log_softmax(logits * self.scale, dim=1)  # [n, c, p]
            one_hot_labels = self.label2one_hot(
                labels, c).unsqueeze(2).repeat(1, 1, p)  # [n, c, p]
            loss = self.compute_loss(log_preds, one_hot_labels)
            self.info.update({'QACEloss': loss.detach().clone()})
            if self.log_accuracy:
                pred = logits.argmax(dim=1)  # [n, p]
                accu = (pred == labels.unsqueeze(1)).float().mean()
                self.info.update({'accuracy': accu})
            if self.log_margin:
                self.info.update({'g_angular_min': torch.min(g_angular).detach().clone()})
                self.info.update({'g_angular_mean': torch.mean(g_angular).detach().clone()})
                self.info.update({'g_angular_max': torch.max(g_angular).detach().clone()})
                self.info.update({'g_add_min': torch.min(g_add).detach().clone()})
                self.info.update({'g_add_mean': torch.mean(g_add).detach().clone()})
                self.info.update({'g_add_max': torch.max(g_add).detach().clone()})
                self.info.update({'margin_scaler_min': torch.min(margin_scaler).detach().clone()})
                self.info.update({'margin_scaler_mean': torch.mean(margin_scaler).detach().clone()})
                self.info.update({'margin_scaler_max': torch.max(margin_scaler).detach().clone()})
                self.info.update({'feat_mean': mean})
                self.info.update({'feat_std': std})
        else:
            log_preds = F.log_softmax(logits * self.scale, dim=1)  # [n, c, p]
            one_hot_labels = self.label2one_hot(
                labels, c).unsqueeze(2).repeat(1, 1, p)  # [n, c, p]
            loss = self.compute_loss(log_preds, one_hot_labels)
            self.info.update({'CEloss': loss.detach().clone()})
            if self.log_accuracy:
                pred = logits.argmax(dim=1)  # [n, p]
                accu = (pred == labels.unsqueeze(1)).float().mean()
                self.info.update({'accuracy': accu})

        return loss, self.info
    
    def compute_loss(self, predis, labels):
        softmax_loss = -(labels * predis).sum(1)  # [n, p]
        losses = softmax_loss.mean(0)   # [p]

        if self.label_smooth:
            smooth_loss = - predis.mean(dim=1)  # [n, p]
            smooth_loss = smooth_loss.mean(0)  # [p]
            losses = smooth_loss * self.eps_smooth + losses * (1. - self.eps_smooth)

        return losses
    
    def label2one_hot(self, label, class_num):
        label = label.unsqueeze(-1)
        batch_size = label.size(0)
        device = label.device
        return torch.zeros(batch_size, class_num).to(device).scatter(1, label, 1)