import torch
import torch.nn as nn

from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.fusion_part.Frequency import Frequency_based_Token_Selection
from modeling.fusion_part.OCFR import OCFR
from modeling.fusion_part.HS_FACSS import HSFACSS
from modeling.fusion_part.AGF import AGF
from modeling.fusion_part.HSL import HSLModule


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.token_dim = 768
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.token_dim = 512
        if 'edge' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.token_dim = 384
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.token_dim = 384
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        num_classes=num_classes,
                                                        camera=camera_num, view=0,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

    def forward(self, x, cam_label, view_label=None):
        cash_x, attn = self.base(x, camera_id=cam_label, view_id=view_label)
        return cash_x, attn

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        model_dict = self.state_dict()
        loaded, skipped = [], []
        for k, v in param_dict.items():
            clean_key = k.replace('module.', '')
            if clean_key not in model_dict:
                skipped.append(clean_key)
                continue
            try:
                model_dict[clean_key].copy_(v)
                loaded.append(clean_key)
            except Exception as e:
                skipped.append(clean_key)
                print('WARNING: skip key {}: checkpoint {} vs model {} ({})'.format(
                    clean_key, v.shape, model_dict[clean_key].shape, e))
        print('Loading pretrained model from {}'.format(trained_path))
        print('  Loaded {}/{} keys'.format(len(loaded), len(model_dict)))
        if skipped:
            print('  Skipped keys: {}'.format(skipped))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        model_dict = self.state_dict()
        for k, v in param_dict.items():
            if k in model_dict:
                try:
                    model_dict[k].copy_(v)
                except Exception as e:
                    print('WARNING: skip key {}: {} ({})'.format(k, e, v.shape))
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class EDITOR(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, factory):
        super(EDITOR, self).__init__()
        # Three Modalities share the same backbone
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, factory)
        self.num_patches = int(cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0]) * int(
            cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1])
        self.HS_FACSS = HSFACSS(dim=self.BACKBONE.token_dim, cfg=cfg)
        self.FREQ_INDEX = Frequency_based_Token_Selection(keep=cfg.MODEL.FREQUENCY_KEEP,
                                                          stride=cfg.MODEL.STRIDE_SIZE[0])
        self.use_agf = cfg.MODEL.AGF
        self.use_hsl = cfg.MODEL.HSL
        self.use_ocfr = bool(cfg.MODEL.OCFR)

        if self.use_agf:
            self.AGF = AGF(dim=self.BACKBONE.token_dim, num_heads=cfg.MODEL.AGF_NUM_HEADS)
        if self.use_ocfr:
            self.memory_cls = OCFR(dim=self.BACKBONE.token_dim, num_class=num_classes, momentum=0.8)

        # High-Order Structure Learning (shared across three modalities)
        if self.use_hsl:
            feat_h = int(cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0])
            feat_w = int(cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1])
            self.HSL = HSLModule(in_features=self.BACKBONE.token_dim,
                                 edges=cfg.MODEL.HSL_EDGES,
                                 filters=cfg.MODEL.HSL_FILTERS,
                                 feat_h=feat_h, feat_w=feat_w,
                                 group_size=cfg.MODEL.HSL_GROUP_SIZE,
                                 graphw=cfg.MODEL.HSL_GRAPHW,
                                 theta1=cfg.MODEL.HSL_THETA1)

        # The output learning params of fused features
        self.FUSE_HEAD = nn.Linear(3 * self.BACKBONE.token_dim, num_classes, bias=False)
        self.FUSE_BN = nn.BatchNorm1d(3 * self.BACKBONE.token_dim)
        self.FUSE_HEAD.apply(weights_init_classifier)

        # The output learning params of RGB/NIR/TIR cls tokens
        self.BACKBONE_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.BACKBONE_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.BACKBONE_HEAD.apply(weights_init_classifier)
        # Here, you can choose to use different head for different modalities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.BACKBONE_HEAD_2 = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        # self.BACKBONE_BN_2 = nn.BatchNorm1d(self.BACKBONE.token_dim)
        # self.BACKBONE_HEAD_2.apply(weights_init_classifier)
        # self.BACKBONE_HEAD_3 = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        # self.BACKBONE_BN_3 = nn.BatchNorm1d(self.BACKBONE.token_dim)
        # self.BACKBONE_HEAD_3.apply(weights_init_classifier)
        # If you use above head, you need to change the forward function to return the scores of different modalities
        # RGB_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(RGB_cls4tri))
        # NIR_cls_score = self.BACKBONE_HEAD_2(self.BACKBONE_BN_2(NIR_cls4tri))
        # TIR_cls_score = self.BACKBONE_HEAD_3(self.BACKBONE_BN_3(TIR_cls4tri))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # In fact, you can choose the AL setting like TOP-ReID, here is the head for AL setting.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.AL = cfg.MODEL.AL
        if self.AL:
            self.AL_HEAD = nn.Linear(3 * self.BACKBONE.token_dim, num_classes, bias=False)
            self.AL_BN = nn.BatchNorm1d(3 * self.BACKBONE.token_dim)
            self.AL_HEAD.apply(weights_init_classifier)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        model_dict = self.state_dict()
        loaded, skipped = [], []
        for k, v in param_dict.items():
            clean_key = k.replace('module.', '')
            if clean_key not in model_dict:
                skipped.append(clean_key)
                continue
            try:
                model_dict[clean_key].copy_(v)
                loaded.append(clean_key)
            except Exception as e:
                skipped.append(clean_key)
                print('WARNING: skip key {}: checkpoint {} vs model {} ({})'.format(
                    clean_key, v.shape, model_dict[clean_key].shape, e))
        missing_keys = [k for k in model_dict if k not in loaded]
        print('Loading pretrained model from {}'.format(trained_path))
        print('  Loaded {}/{} keys'.format(len(loaded), len(model_dict)))
        if skipped:
            print('  Skipped keys (in checkpoint but not loaded): {}'.format(skipped))
        if missing_keys:
            print('  Missing keys (randomly initialized): {}'.format(missing_keys))

    def forward(self, x, cam_label=None, label=None, view_label=None, img_path=None, mode=1,
                writer=None, epoch=None):
        if self.training:
            RGB = x['RGB']
            NIR = x['NI']
            TIR = x['TI']
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=TIR, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
            TIR_feat, TIR_attn = self.BACKBONE(TIR, cam_label=cam_label, view_label=view_label)

            # HSL: enhance patch tokens with high-order structure learning
            if self.use_hsl:
                RGB_feat = torch.cat([RGB_feat[:, :1, :], self.HSL(RGB_feat[:, 1:, :])], dim=1)
                NIR_feat = torch.cat([NIR_feat[:, :1, :], self.HSL(NIR_feat[:, 1:, :])], dim=1)
                TIR_feat = torch.cat([TIR_feat[:, :1, :], self.HSL(TIR_feat[:, 1:, :])], dim=1)

            RGB_cls4tri = RGB_feat[:, 0, :]
            NIR_cls4tri = NIR_feat[:, 0, :]
            TIR_cls4tri = TIR_feat[:, 0, :]
            if self.AL:
                ori = torch.cat([RGB_cls4tri, NIR_cls4tri, TIR_cls4tri], dim=-1)
                ori_score = self.AL_HEAD(self.AL_BN(ori))
            else:
                RGB_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(RGB_cls4tri))
                NIR_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(NIR_cls4tri))
                TIR_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(TIR_cls4tri))

            RGB_feat_s, NIR_feat_s, TIR_feat_s, mask = self.HS_FACSS(RGB_feat=RGB_feat,
                                                                     RGB_attn=RGB_attn,
                                                                     NIR_feat=NIR_feat,
                                                                     NIR_attn=NIR_attn,
                                                                     TIR_feat=TIR_feat,
                                                                     TIR_attn=TIR_attn,
                                                                     img_path=img_path,
                                                                     epoch=epoch, writer=writer,
                                                                     mask_fre=mask_fre)

            if self.use_agf:
                cls4t = self.AGF(RGB_feat_s, NIR_feat_s, TIR_feat_s)
            else:
                cls4t = torch.cat([RGB_feat_s[:, 0, :], NIR_feat_s[:, 0, :], TIR_feat_s[:, 0, :]], dim=-1)
            if self.use_ocfr:
                RGB_cls = RGB_feat_s[:, 0, :]
                NIR_cls = NIR_feat_s[:, 0, :]
                TIR_cls = TIR_feat_s[:, 0, :]
                loss_aux = self.memory_cls(RGB_cls, NIR_cls, TIR_cls, label, epoch=epoch)
            else:
                loss_aux = torch.zeros((), device=RGB_feat.device)
            score = self.FUSE_HEAD(self.FUSE_BN(cls4t))
            if self.AL:
                return score, cls4t, ori_score, ori, loss_aux
            else:
                return score, cls4t, RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, TIR_cls_score, TIR_cls4tri, loss_aux
        else:
            RGB = x['RGB']
            NIR = x['NI']
            TIR = x['TI']
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=TIR, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
            TIR_feat, TIR_attn = self.BACKBONE(TIR, cam_label=cam_label, view_label=view_label)

            # HSL: enhance patch tokens with high-order structure learning
            if self.use_hsl:
                RGB_feat = torch.cat([RGB_feat[:, :1, :], self.HSL(RGB_feat[:, 1:, :])], dim=1)
                NIR_feat = torch.cat([NIR_feat[:, :1, :], self.HSL(NIR_feat[:, 1:, :])], dim=1)
                TIR_feat = torch.cat([TIR_feat[:, :1, :], self.HSL(TIR_feat[:, 1:, :])], dim=1)

            RGB_feat_s, NIR_feat_s, TIR_feat_s, mask = self.HS_FACSS(RGB_feat=RGB_feat,
                                                                     RGB_attn=RGB_attn,
                                                                     NIR_feat=NIR_feat,
                                                                     NIR_attn=NIR_attn,
                                                                     TIR_feat=TIR_feat,
                                                                     TIR_attn=TIR_attn,
                                                                     img_path=img_path,
                                                                     epoch=epoch, writer=writer,
                                                                     mask_fre=mask_fre)

            if self.use_agf:
                cls4t = self.AGF(RGB_feat_s, NIR_feat_s, TIR_feat_s)
            else:
                cls4t = torch.cat([RGB_feat_s[:, 0, :], NIR_feat_s[:, 0, :], TIR_feat_s[:, 0, :]], dim=-1)
            return cls4t

    def forward_two_modalities(self, x, cam_label=None, label=None, view_label=None, cross_type=None, img_path=None,
                               mode=1,
                               writer=None, epoch=None):
        # This forward function is used for the two modalities datasets like RGBN300
        if self.training:
            RGB = x['RGB']
            NIR = x['NI']
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=None, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)

            # HSL: enhance patch tokens with high-order structure learning
            if self.use_hsl:
                RGB_feat = torch.cat([RGB_feat[:, :1, :], self.HSL(RGB_feat[:, 1:, :])], dim=1)
                NIR_feat = torch.cat([NIR_feat[:, :1, :], self.HSL(NIR_feat[:, 1:, :])], dim=1)

            RGB_cls4tri = RGB_feat[:, 0, :]
            NIR_cls4tri = NIR_feat[:, 0, :]
            # Here, you need to change the head for the AL setting to 2*token_dim
            if self.AL:
                ori = torch.cat([RGB_cls4tri, NIR_cls4tri], dim=-1)
                ori_score = self.AL_HEAD(self.AL_BN(ori))
            else:
                RGB_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(RGB_cls4tri))
                NIR_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(NIR_cls4tri))

            RGB_feat_s, NIR_feat_s, mask = self.HS_FACSS(RGB_feat=RGB_feat,
                                                         RGB_attn=RGB_attn,
                                                         NIR_feat=NIR_feat,
                                                         NIR_attn=NIR_attn,
                                                         TIR_feat=None,
                                                         TIR_attn=None,
                                                         img_path=img_path,
                                                         epoch=epoch, writer=writer,
                                                         mask_fre=mask_fre)

            TIR_feat_s = torch.zeros_like(RGB_feat_s)
            if self.use_agf:
                cls4t = self.AGF(RGB_feat_s, NIR_feat_s, TIR_feat_s)
            else:
                cls4t = torch.cat([RGB_feat_s[:, 0, :], NIR_feat_s[:, 0, :], TIR_feat_s[:, 0, :]], dim=-1)
            if self.use_ocfr:
                RGB_cls = RGB_feat_s[:, 0, :]
                NIR_cls = NIR_feat_s[:, 0, :]
                TIR_cls = TIR_feat_s[:, 0, :]
                loss_aux = self.memory_cls(RGB_cls, NIR_cls, TIR_cls, label, epoch=epoch)
            else:
                loss_aux = torch.zeros((), device=RGB_feat.device)
            score = self.FUSE_HEAD(self.FUSE_BN(cls4t))
            if self.AL:
                return score, cls4t, ori_score, ori, loss_aux
            else:
                return score, cls4t, RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, loss_aux

        else:
            RGB = x['RGB']
            NIR = x['NI']
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=None, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)

            # HSL: enhance patch tokens with high-order structure learning
            if self.use_hsl:
                RGB_feat = torch.cat([RGB_feat[:, :1, :], self.HSL(RGB_feat[:, 1:, :])], dim=1)
                NIR_feat = torch.cat([NIR_feat[:, :1, :], self.HSL(NIR_feat[:, 1:, :])], dim=1)

            RGB_feat_s, NIR_feat_s, mask = self.HS_FACSS(RGB_feat=RGB_feat,
                                                         RGB_attn=RGB_attn,
                                                         NIR_feat=NIR_feat,
                                                         NIR_attn=NIR_attn,
                                                         TIR_feat=None,
                                                         TIR_attn=None,
                                                         img_path=img_path,
                                                         epoch=epoch, writer=writer,
                                                         mask_fre=mask_fre)

            TIR_feat_s = torch.zeros_like(RGB_feat_s)
            if self.use_agf:
                cls4t = self.AGF(RGB_feat_s, NIR_feat_s, TIR_feat_s)
            else:
                cls4t = torch.cat([RGB_feat_s[:, 0, :], NIR_feat_s[:, 0, :], TIR_feat_s[:, 0, :]], dim=-1)
            return cls4t


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
}


def make_model(cfg, num_class, camera_num):
    model = EDITOR(num_class, cfg, camera_num, __factory_T_type)
    print('===========Building EDITOR===========')
    return model
