import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class QATriplet(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0, log_margin=False):
        super(QATriplet, self).__init__(loss_term_weight)
        self.margin = margin
        self.log_margin = log_margin

    @gather_and_scale_wrapper
    def forward(self, feature_hpp, embeddings, labels):
        # feature_hpp: [n, c, p]
        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        
        ap_margin, an_margin = self.QualityAdaptiveMargin(feature_hpp, labels, ref_label, self.margin)

        dist_diff = ((ap_dist + ap_margin) - (an_dist - an_margin)).view(dist.size(0), -1)

        loss = F.relu(dist_diff)
        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'QATriplet_loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})
        if self.log_margin:
            self.info.update({'ap_margin_min': torch.min(ap_margin).detach().clone()})
            self.info.update({'ap_margin_mean': torch.mean(ap_margin).detach().clone()})
            self.info.update({'ap_margin_max': torch.max(ap_margin).detach().clone()})
            self.info.update({'an_margin_min': torch.min(an_margin).detach().clone()})
            self.info.update({'an_margin_mean': torch.mean(an_margin).detach().clone()})
            self.info.update({'an_margin_max': torch.max(an_margin).detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist
    
    def QualityAdaptiveMargin(self, feature_hpp, row_labels, clo_label, margin, eps_adaFace=1e-4, h=0.333):
        """
            feature_hpp: tensor with size [n, c, p]
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        feature_norm = torch.norm(feature_hpp, p=2, dim=1, keepdim=False)  # [n, p]
        feature_norm = torch.mean(feature_norm, dim=-1, keepdim=False)    # [n]
        safe_feature_norms = torch.clip(feature_norm, min=0.001, max=100) # for stability   # [n]
        safe_feature_norms = safe_feature_norms.clone().detach()          # [n]

        with torch.no_grad():
            mean = safe_feature_norms.mean().detach()  # [1]
            std = safe_feature_norms.std().detach()  # [1]
        
        margin_scaler = (safe_feature_norms - mean) / (std + eps_adaFace) # 66% between -1, 1
        margin_scaler = margin_scaler * h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)  # [n]
        """ Strategy - 1 """
        AdaMargin = 1.5 * margin + (0.5 * margin * margin_scaler)  # [n], g_angular∈[m, 2m]
        AdaMarginMatrix = torch.min(AdaMargin.unsqueeze(1), AdaMargin.unsqueeze(0))  # Min Quality

        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        n, _ = AdaMarginMatrix.size()
        ap_margin = AdaMarginMatrix[matches].view(n, -1, 1)
        an_margin = AdaMarginMatrix[diffenc].view(n, 1, -1)

        return ap_margin, an_margin