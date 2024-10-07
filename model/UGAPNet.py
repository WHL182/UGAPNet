import torch
from torch import nn
import torch.nn.functional as F
from Semi_supervise.model.ASPP import ASPP
from Semi_supervise.model.PSPNet import OneModel as PSPNet
import numpy as np

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.un_shot = args.un_shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.print_freq = args.print_freq / 2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60

        assert self.layers in [50, 101, 152]

        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
        weight_path = './initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split, backbone_str)
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:  # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)
        # Meta Learner
        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def get_optim(self, model, args, LR):

        optimizer = torch.optim.SGD(
            [
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},
                {'params': model.cls_meta.parameters()},
                {'params': model.alpha},
            ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)

        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False

    def Prior_Mask(self,final_supp_list,mask_list, query_feat_4, query_feat_3):
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear',
                                     align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]),
                                       mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)

        return corr_query_mask

    def Pseudo_lable_Prediction_Module(self, x, s_x, s_y):
        x_size = x.size()
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)
        # Support Feature
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_pro = Weighted_GAP(supp_feat, mask)
            supp_pro_list.append(supp_pro)
        # Prior Mask
        corr_query_mask = self.Prior_Mask(final_supp_list, mask_list, query_feat_4, query_feat_3)
        corr_query_mask = torch.mean(corr_query_mask, dim=1, keepdim=True)
        # Support Prototype
        supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
        supp_pro = torch.mean(supp_pro, dim=2, keepdim=True)
        # Tile & Cat
        concat_feat = supp_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, concat_feat, corr_query_mask], 1)  # 256+256+1
        merge_feat = self.init_merge(merge_feat)
        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)  # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
        meta_out = meta_out.softmax(1)
        pseudo_mask = torch.where(meta_out[:, 1:, :, :] > 0.5, 1.0, 0.0)

        return pseudo_mask

    def Prototype_Rectification_Module(self, x_u, x_u_mask, query_feat_4, supp_pro, corr_query_mask):
        # Support Feature
        pseudo_pro_list = []
        pseudo_supp_list = []
        pseudo_mask_list = []
        for i in range(self.un_shot):
            pseudo_mask = (x_u_mask[:, i, :, :] == 1).float().unsqueeze(1)
            pseudo_mask_list.append(pseudo_mask)
            with torch.no_grad():
                pseudo_feat_0 = self.layer0(x_u[:, i, :, :, :])
                pseudo_feat_1 = self.layer1(pseudo_feat_0)
                pseudo_feat_2 = self.layer2(pseudo_feat_1)
                pseudo_feat_3 = self.layer3(pseudo_feat_2)
                pseudo_mask = F.interpolate(pseudo_mask, size=(pseudo_feat_3.size(2), pseudo_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                pseudo_feat_4 = self.layer4(pseudo_feat_3 * pseudo_mask)
                pseudo_supp_list.append(pseudo_feat_4)
                if self.vgg:
                    pseudo_feat_2 = F.interpolate(pseudo_feat_2, size=(pseudo_feat_3.size(2), pseudo_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)

            pseudo_feat = torch.cat([pseudo_feat_3, pseudo_feat_2], 1)
            pseudo_feat = self.down_supp(pseudo_feat)############################
            pseudo_pro = Weighted_GAP(pseudo_feat, pseudo_mask)
            pseudo_pro_list.append(pseudo_pro)
        # Prior Similarity Mask
        pseudo_corr_query_mask = self.Prior_Mask(pseudo_supp_list, pseudo_mask_list, query_feat_4, pseudo_feat_3)

        pseudo_supp_pro = torch.stack(pseudo_pro_list, dim=1)
        supp_pro_ = supp_pro.unsqueeze(1)
        deta_pro = nn.CosineSimilarity(2)(pseudo_supp_pro, supp_pro_)
        deta_pro_soft = deta_pro.softmax(1)
        un_supp_pro_final = deta_pro_soft.unsqueeze(-1) * pseudo_supp_pro
        un_supp_pro_final = torch.mean(un_supp_pro_final, dim=1)
        un_corr_query_mask_final = deta_pro_soft * pseudo_corr_query_mask
        un_corr_query_mask_final = torch.mean(un_corr_query_mask_final, dim=1, keepdim=True)
        revise_pro = un_supp_pro_final
        revise_mask = un_corr_query_mask_final
        if self.alpha == -1:
            adaptive_supp_pro = supp_pro
            corr_query_mask = corr_query_mask
        else:
            adaptive_supp_pro = (supp_pro + self.alpha * revise_pro) / (1 + self.alpha)
            corr_query_mask = (corr_query_mask + self.alpha * revise_mask) / (1 + self.alpha)

        return adaptive_supp_pro, corr_query_mask

    def Uncertainty_Estimation_Module(self, y_m, query_feat_3, query_feat, supp_pro, bs):
        mask_uncertainty = y_m.float().unsqueeze(1)
        mask_uncertainty = F.interpolate(mask_uncertainty, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                         mode='bilinear', align_corners=True)
        query_feat = F.interpolate(query_feat, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                         mode='bilinear', align_corners=True)
        query_pro = Weighted_GAP(query_feat, mask_uncertainty)
        sup_pro_loss = supp_pro
        deta_pro_loss = nn.CosineSimilarity(1)(query_pro, sup_pro_loss).squeeze(-1).squeeze(-1)
        aux_loss1 = 0
        for k in range(bs):
            aux_loss1 += -torch.log((1 - deta_pro_loss[k]) / 2)
        aux_loss1 = aux_loss1 / bs

        return aux_loss1

    def forward(self, x, s_x, s_u, s_y, y_m, y_b, cat_idx=None):
        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        # Support Feature
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_pro = Weighted_GAP(supp_feat, mask)
            supp_pro_list.append(supp_pro)

        # Prior Similarity Mask
        corr_query_mask = self.Prior_Mask(final_supp_list, mask_list, query_feat_4, query_feat_3)
        corr_query_mask = torch.mean(corr_query_mask, dim=1, keepdim=True)

        # Support Prototype
        supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
        supp_pro = torch.mean(supp_pro, dim=2, keepdim=True)
        # Pseudo lable
        mask_s_u_list = []
        for i in range(self.un_shot):
            s_u_each = s_u[:, i, :, :, :]
            mask_s_u = self.Pseudo_lable_Prediction_Module(s_u_each, s_x, s_y)
            mask_s_u_list.append(mask_s_u)
        mask_u_s = torch.cat(mask_s_u_list, dim=1)
        # Prototype Rectification
        adaptive_supp_pro, corr_query_mask = self.Prototype_Rectification_Module(s_u, mask_u_s, query_feat_4, supp_pro, corr_query_mask)

        # Tile & Cat
        concat_feat = adaptive_supp_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, concat_feat, corr_query_mask], 1)  # 256+256+1
        merge_feat = self.init_merge(merge_feat)

        # Base and Meta
        base_out = self.learner_base(query_feat_4)

        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)  # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)

        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            query_feat = F.interpolate(query_feat, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            main_loss = self.criterion(meta_out, y_m.long())
            # Uncertainty Estimation
            aux_loss1 = self.Uncertainty_Estimation_Module(y_m, query_feat_3, query_feat, supp_pro, bs)
            aux_loss2 = self.criterion(base_out, y_b.long())
            return meta_out.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return meta_out, base_out, query_feat

