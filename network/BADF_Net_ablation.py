"""
=============================================================
BADF-Net -- Configurable Version for Architectural Ablation
-------------------------------------------------------------
Every novel module (BADA, BCE, CSBF, Boundary Gate, Skip
Attention, SE blocks, backbone) is wrapped behind a flag in
AblationConfig, so a single model class can instantiate any
ablation variant (A0-A11) just by changing the config.
=============================================================
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =========================================================================
# Ablation configuration
# =========================================================================

@dataclass
class AblationConfig:
    use_bada: bool = True                 # A1 removes this
    use_bce: bool = True                  # A2 removes this (only matters if use_bada=True)
    csbf_mode: str = 'adaptive'           # 'adaptive' | 'equal' | 'concat'  (A3, A4)
    use_boundary_gate: bool = True        # A5 removes this
    skip_mode: str = 'attention'          # 'attention' | 'concat'          (A6)
    use_se: bool = True                   # A7 removes SE blocks everywhere
    attention_mode: str = 'both'          # 'both' | 'channel' | 'spatial' | 'none'  (A8, inside BADA)
    boundary_conv_type: str = 'depthwise' # 'depthwise' | 'standard'        (A9)
    encoder_name: str = 'resnet34'        # 'resnet34' | 'resnet18'         (A10)
    use_residual_decoder: bool = True     # A11 removes residual refinement


# Registry: ablation_id -> (description, config). Iterate this in the runner script.
ABLATION_CONFIGS = {
    "A0_full_model":            ("Full BADF-Net (baseline)",                       AblationConfig()),
    "A1_no_BADA":                ("Remove Boundary-Aware Dynamic Attention",        AblationConfig(use_bada=False)),
    "A2_no_BCE":                 ("Remove Boundary Confidence Estimation",          AblationConfig(use_bce=False)),
    "A3_CSBF_equal_weights":     ("CSBF with fixed equal scale weights",            AblationConfig(csbf_mode='equal')),
    "A4_CSBF_concat":            ("CSBF replaced with concat + 1x1 conv",           AblationConfig(csbf_mode='concat')),
    "A5_no_boundary_gate":       ("Remove decoder Boundary Gate",                   AblationConfig(use_boundary_gate=False)),
    "A6_skip_concat":            ("Skip fusion: plain concat (no attention gate)",  AblationConfig(skip_mode='concat')),
    "A7_no_SE":                  ("Remove all SE blocks",                          AblationConfig(use_se=False)),
    "A8_channel_attention_only": ("BADA: channel attention only",                  AblationConfig(attention_mode='channel')),
    "A8_spatial_attention_only": ("BADA: spatial attention only",                  AblationConfig(attention_mode='spatial')),
    "A8_no_attention":           ("BADA: no channel/spatial attention",            AblationConfig(attention_mode='none')),
    "A9_standard_conv":          ("Boundary extraction: standard conv (no depthwise-separable)", AblationConfig(boundary_conv_type='standard')),
    "A10_resnet18_backbone":     ("Lighter backbone: ResNet18 instead of ResNet34", AblationConfig(encoder_name='resnet18')),
    "A11_no_residual_decoder":   ("Remove residual refinement in decoder",          AblationConfig(use_residual_decoder=False)),
}


# =========================================================================
# Basic building blocks (unchanged from the original architecture)
# =========================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, 1, 1),
            ConvBNReLU(out_channels, out_channels, 3, 1, 1))

    def forward(self, x):
        return self.conv(x)


class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def se_or_identity(channels, use_se):
    return SEBlock(channels) if use_se else nn.Identity()


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


# =========================================================================
# Boundary-Aware Dynamic Attention (configurable)
# =========================================================================

class BoundaryAwareDynamicAttention(nn.Module):
    def __init__(self, channels, cfg: AblationConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.boundary_conv_type == 'depthwise':
            self.boundary_conv = nn.Sequential(DepthwiseSeparableConv(channels, channels), ConvBNReLU(channels, channels))
        else:
            self.boundary_conv = nn.Sequential(ConvBNReLU(channels, channels), ConvBNReLU(channels, channels))

        self.channel_attention = ChannelAttention(channels) if cfg.attention_mode in ('both', 'channel') else None
        self.spatial_attention = SpatialAttention() if cfg.attention_mode in ('both', 'spatial') else None

        self.fusion = nn.Sequential(ConvBNReLU(channels, channels), se_or_identity(channels, cfg.use_se))

    def forward(self, x):
        boundary = self.boundary_conv(x)
        if self.channel_attention is not None:
            boundary = boundary * self.channel_attention(boundary)
        if self.spatial_attention is not None:
            boundary = boundary * self.spatial_attention(boundary)
        refined = x + boundary
        refined = self.fusion(refined)
        return refined, boundary


class BoundaryConfidenceEstimation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.confidence = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, boundary_feature):
        confidence = self.sigmoid(self.confidence(boundary_feature))
        return boundary_feature * confidence, confidence


# =========================================================================
# Cross-Scale Boundary Fusion (configurable fusion mode)
# =========================================================================

class FeatureProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.project(x)


class ScaleWeightGenerator(nn.Module):
    def __init__(self, channels, num_scales):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * num_scales, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_scales))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_list):
        pooled = torch.cat([self.pool(f).flatten(1) for f in feature_list], dim=1)
        return self.softmax(self.fc(pooled))


class CrossScaleBoundaryFusion(nn.Module):
    def __init__(self, cfg: AblationConfig, in_channels=(64, 64, 128, 256), out_channels=256):
        super().__init__()
        self.cfg = cfg
        c1, c2, c3, c4 = in_channels
        self.p1 = FeatureProjection(c1, out_channels)
        self.p2 = FeatureProjection(c2, out_channels)
        self.p3 = FeatureProjection(c3, out_channels)
        self.p4 = FeatureProjection(c4, out_channels)

        if cfg.csbf_mode == 'adaptive':
            self.weight_generator = ScaleWeightGenerator(out_channels, 4)
        elif cfg.csbf_mode == 'concat':
            self.concat_reduce = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)

        self.refine = nn.Sequential(
            ConvBNReLU(out_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            se_or_identity(out_channels, cfg.use_se))

    def forward(self, b1, b2, b3, b4):
        b1, b2, b3, b4 = self.p1(b1), self.p2(b2), self.p3(b3), self.p4(b4)
        target_size = b4.size()[2:]
        b1 = F.interpolate(b1, size=target_size, mode='bilinear', align_corners=False)
        b2 = F.interpolate(b2, size=target_size, mode='bilinear', align_corners=False)
        b3 = F.interpolate(b3, size=target_size, mode='bilinear', align_corners=False)

        if self.cfg.csbf_mode == 'adaptive':
            weights = self.weight_generator([b1, b2, b3, b4])
            w1, w2, w3, w4 = [weights[:, i].view(-1, 1, 1, 1) for i in range(4)]
            fusion = w1 * b1 + w2 * b2 + w3 * b3 + w4 * b4
        elif self.cfg.csbf_mode == 'equal':
            fusion = 0.25 * (b1 + b2 + b3 + b4)
        else:  # 'concat'
            fusion = self.concat_reduce(torch.cat([b1, b2, b3, b4], dim=1))

        return self.refine(fusion)


# =========================================================================
# Decoder-side modules (configurable gate / skip mode / residual)
# =========================================================================

class BoundaryGate(nn.Module):
    def __init__(self, decoder_channels, boundary_channels):
        super().__init__()
        self.boundary_proj = nn.Sequential(
            nn.Conv2d(boundary_channels, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True))
        self.attention = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, decoder_feature, boundary_feature):
        if boundary_feature.shape[2:] != decoder_feature.shape[2:]:
            boundary_feature = F.interpolate(boundary_feature, size=decoder_feature.shape[2:], mode='bilinear', align_corners=False)
        boundary_feature = self.boundary_proj(boundary_feature)
        gate = self.attention(boundary_feature)
        return decoder_feature * gate + decoder_feature


class SkipAttentionGate(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, inter_channels):
        super().__init__()
        self.theta = nn.Sequential(nn.Conv2d(encoder_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.phi = nn.Sequential(nn.Conv2d(decoder_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, encoder_feature, decoder_feature):
        if decoder_feature.shape[2:] != encoder_feature.shape[2:]:
            decoder_feature = F.interpolate(decoder_feature, size=encoder_feature.shape[2:], mode='bilinear', align_corners=False)
        attention = self.psi(self.relu(self.theta(encoder_feature) + self.phi(decoder_feature)))
        return encoder_feature * attention


class AdaptiveSkipFusion(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels, cfg: AblationConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.skip_mode == 'attention':
            self.attention = SkipAttentionGate(encoder_channels, decoder_channels, out_channels)
        self.fusion = nn.Sequential(
            ConvBNReLU(encoder_channels + decoder_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            se_or_identity(out_channels, cfg.use_se))

    def forward(self, encoder_feature, decoder_feature):
        if self.cfg.skip_mode == 'attention':
            encoder_feature = self.attention(encoder_feature, decoder_feature)
        if decoder_feature.shape[2:] != encoder_feature.shape[2:]:
            decoder_feature = F.interpolate(decoder_feature, size=encoder_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([encoder_feature, decoder_feature], dim=1))


class BoundaryGuidedFusion(nn.Module):
    def __init__(self, channels, boundary_channels, cfg: AblationConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_boundary_gate:
            self.boundary_gate = BoundaryGate(channels, boundary_channels)
        refine_block = ResidualConv(channels, channels) if cfg.use_residual_decoder else DoubleConv(channels, channels)
        self.refine = nn.Sequential(refine_block, se_or_identity(channels, cfg.use_se))

    def forward(self, decoder_feature, boundary_feature):
        if self.cfg.use_boundary_gate:
            decoder_feature = self.boundary_gate(decoder_feature, boundary_feature)
        return self.refine(decoder_feature)


class AdaptiveDecoderBlock(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels, cfg: AblationConfig, boundary_channels=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.skip_fusion = AdaptiveSkipFusion(encoder_channels, decoder_channels, out_channels, cfg)
        self.boundary_fusion = BoundaryGuidedFusion(out_channels, boundary_channels, cfg)
        refine_block = ResidualConv(out_channels, out_channels) if cfg.use_residual_decoder else DoubleConv(out_channels, out_channels)
        self.refine = nn.Sequential(refine_block, ConvBNReLU(out_channels, out_channels), se_or_identity(out_channels, cfg.use_se))

    def forward(self, decoder_feature, encoder_feature, boundary_feature):
        decoder_feature = self.up(decoder_feature)
        decoder_feature = self.skip_fusion(encoder_feature, decoder_feature)
        decoder_feature = self.boundary_fusion(decoder_feature, boundary_feature)
        return self.refine(decoder_feature)


class DecoderHead(nn.Module):
    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.decoder4 = AdaptiveDecoderBlock(256, 512, 256, cfg)
        self.decoder3 = AdaptiveDecoderBlock(128, 256, 128, cfg)
        self.decoder2 = AdaptiveDecoderBlock(64, 128, 64, cfg)
        self.decoder1 = AdaptiveDecoderBlock(64, 64, 64, cfg)

    def forward(self, e1, e2, e3, e4, bottleneck, boundary_feature):
        d4 = self.decoder4(bottleneck, e4, boundary_feature)
        d3 = self.decoder3(d4, e3, boundary_feature)
        d2 = self.decoder2(d3, e2, boundary_feature)
        d1 = self.decoder1(d2, e1, boundary_feature)
        return d1


class SegmentationHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=6):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, 64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, kernel_size=1))

    def forward(self, x):
        return self.head(x)


# =========================================================================
# Encoder (configurable backbone + optional BADA/BCE)
# =========================================================================

class ResNetEncoder(nn.Module):
    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.encoder_name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if cfg.use_bada:
            self.bada1 = BoundaryAwareDynamicAttention(64, cfg)
            self.bada2 = BoundaryAwareDynamicAttention(64, cfg)
            self.bada3 = BoundaryAwareDynamicAttention(128, cfg)
            self.bada4 = BoundaryAwareDynamicAttention(256, cfg)

            if cfg.use_bce:
                self.bce1 = BoundaryConfidenceEstimation(64)
                self.bce2 = BoundaryConfidenceEstimation(64)
                self.bce3 = BoundaryConfidenceEstimation(128)
                self.bce4 = BoundaryConfidenceEstimation(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        e1 = self.relu(x)
        x = self.maxpool(e1)
        e2 = self.layer1(x)
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        bottleneck = self.layer4(e4)

        if self.cfg.use_bada:
            b1, _ = self.bada1(e1)
            b2, _ = self.bada2(e2)
            b3, _ = self.bada3(e3)
            b4, _ = self.bada4(e4)
            if self.cfg.use_bce:
                b1, _ = self.bce1(b1)
                b2, _ = self.bce2(b2)
                b3, _ = self.bce3(b3)
                b4, _ = self.bce4(b4)
        else:
            # No boundary attention: raw encoder features stand in for the
            # "boundary" branch fed to CSBF / decoder.
            b1, b2, b3, b4 = e1, e2, e3, e4

        return e1, e2, e3, e4, bottleneck, b1, b2, b3, b4


# =========================================================================
# BADF-Net (configurable)
# =========================================================================

class BADF_Net_Ablation(nn.Module):
    def __init__(self, num_classes=6, cfg: AblationConfig = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else AblationConfig()
        self.encoder = ResNetEncoder(self.cfg)
        self.csbf = CrossScaleBoundaryFusion(self.cfg, in_channels=(64, 64, 128, 256), out_channels=256)
        self.decoder = DecoderHead(self.cfg)
        self.segmentation_head = SegmentationHead(in_channels=64, num_classes=num_classes)

    def forward(self, x):
        e1, e2, e3, e4, bottleneck, b1, b2, b3, b4 = self.encoder(x)
        boundary_feature = self.csbf(b1, b2, b3, b4)
        decoder_feature = self.decoder(e1, e2, e3, e4, bottleneck, boundary_feature)
        output = self.segmentation_head(decoder_feature)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output