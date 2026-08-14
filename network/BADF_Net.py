"""
==========================================================
BADF-Net
Boundary-Aware Dynamic Fusion Network
Part-1
----------------------------------------------------------
Contains:
1. Imports
2. Helper Layers
3. ConvBNReLU
4. ResidualConv
5. SE Channel Recalibration
6. Boundary-Aware Dynamic Attention (BADA)
==========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


##############################################################
# Basic Convolution Block
##############################################################

class ConvBNReLU(nn.Module):
    """
    Conv -> BatchNorm -> ReLU
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


##############################################################
# Double Conv Block
##############################################################

class DoubleConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            ConvBNReLU(in_channels,
                       out_channels,
                       3,
                       1,
                       1),

            ConvBNReLU(out_channels,
                       out_channels,
                       3,
                       1,
                       1)
        )

    def forward(self, x):
        return self.conv(x)


##############################################################
# Residual Convolution
##############################################################

class ResidualConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):

        super().__init__()

        self.conv1 = ConvBNReLU(
            in_channels,
            out_channels)

        self.conv2 = ConvBNReLU(
            out_channels,
            out_channels)

        self.shortcut = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False)

    def forward(self, x):

        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)

        return x + residual


##############################################################
# Depthwise Separable Convolution
##############################################################

class DepthwiseSeparableConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1):

        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)

        return self.relu(x)


##############################################################
# Squeeze-and-Excitation Block
##############################################################

class SEBlock(nn.Module):
    """
    Hu et al.
    Squeeze-and-Excitation Networks (CVPR 2018)
    """

    def __init__(self,
                 channels,
                 reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(
                channels,
                channels // reduction,
                bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(
                channels // reduction,
                channels,
                bias=False),

            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(
            b,
            c,
            1,
            1)

        return x * y.expand_as(x)


##############################################################
# Spatial Attention
##############################################################

class SpatialAttention(nn.Module):

    def __init__(self,
                 kernel_size=7):

        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=padding,
            bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True)

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True)

        attention = torch.cat(
            [avg_out,
             max_out],
            dim=1)

        attention = self.conv(attention)

        return self.sigmoid(attention)


##############################################################
# Channel Attention
##############################################################

class ChannelAttention(nn.Module):

    def __init__(self,
                 channels,
                 reduction=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                channels // reduction,
                1,
                bias=False),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // reduction,
                channels,
                1,
                bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.fc(self.avg_pool(x))

        mx = self.fc(self.max_pool(x))

        return self.sigmoid(avg + mx)


##############################################################
# Boundary-Aware Dynamic Attention (Novel)
##############################################################

class BoundaryAwareDynamicAttention(nn.Module):
    """
    Proposed Module

    Input
        Feature Map

    Output
        Boundary Refined Feature
        Boundary Map
    """

    def __init__(self,
                 channels):

        super().__init__()

        ####################################################
        # Boundary Extraction
        ####################################################

        self.boundary_conv = nn.Sequential(

            DepthwiseSeparableConv(
                channels,
                channels),

            ConvBNReLU(
                channels,
                channels)
        )

        ####################################################
        # Dynamic Channel Attention
        ####################################################

        self.channel_attention = ChannelAttention(
            channels)

        ####################################################
        # Dynamic Spatial Attention
        ####################################################

        self.spatial_attention = SpatialAttention()

        ####################################################
        # Feature Fusion
        ####################################################

        self.fusion = nn.Sequential(

            ConvBNReLU(
                channels,
                channels),

            SEBlock(channels)
        )

    def forward(self, x):

        ###############################################
        # Boundary Feature
        ###############################################

        boundary = self.boundary_conv(x)

        ###############################################
        # Channel Attention
        ###############################################

        ca = self.channel_attention(boundary)

        boundary = boundary * ca

        ###############################################
        # Spatial Attention
        ###############################################

        sa = self.spatial_attention(boundary)

        boundary = boundary * sa

        ###############################################
        # Residual Refinement
        ###############################################

        refined = x + boundary

        ###############################################
        # Final Fusion
        ###############################################

        refined = self.fusion(refined)

        return refined, boundary


##############################################################
# Boundary Confidence Estimation (BCE)
##############################################################

class BoundaryConfidenceEstimation(nn.Module):
    """
    Estimate boundary confidence map.

    Input:
        Boundary Feature

    Output:
        Refined Boundary Feature
        Confidence Map
    """

    def __init__(self, channels):

        super().__init__()

        self.confidence = nn.Sequential(

            nn.Conv2d(
                channels,
                channels // 2,
                kernel_size=3,
                padding=1,
                bias=False),

            nn.BatchNorm2d(channels // 2),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // 2,
                1,
                kernel_size=1,
                bias=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, boundary_feature):

        confidence = self.confidence(boundary_feature)

        confidence = self.sigmoid(confidence)

        refined_boundary = boundary_feature * confidence

        return refined_boundary, confidence


##############################################################
# Feature Projection
##############################################################

class FeatureProjection(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):

        super().__init__()

        self.project = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.project(x)


##############################################################
# Adaptive Scale Weight Generator
##############################################################

class ScaleWeightGenerator(nn.Module):

    def __init__(self,
                 channels,
                 num_scales):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(
                channels * num_scales,
                channels),

            nn.ReLU(inplace=True),

            nn.Linear(
                channels,
                num_scales)
        )

        self.softmax = nn.Softmax(dim=1)

        self.num_scales = num_scales

    def forward(self, feature_list):

        pooled = []

        for feat in feature_list:

            pooled.append(
                self.pool(feat).flatten(1)
            )

        pooled = torch.cat(pooled, dim=1)

        weights = self.fc(pooled)

        weights = self.softmax(weights)

        return weights


##############################################################
# Cross Scale Boundary Fusion (CSBF)
##############################################################
# Better default
class CrossScaleBoundaryFusion(nn.Module):

    def __init__(
            self,
            in_channels=(64, 128, 256, 512),
            out_channels=256):

        super().__init__()

        c1, c2, c3, c4 = in_channels

        #################################################
        # Projection Layers
        #################################################

        self.p1 = FeatureProjection(c1, out_channels)
        self.p2 = FeatureProjection(c2, out_channels)
        self.p3 = FeatureProjection(c3, out_channels)
        self.p4 = FeatureProjection(c4, out_channels)

        #################################################
        # Adaptive Weight Generator
        #################################################

        self.weight_generator = ScaleWeightGenerator(
            out_channels,
            4
        )

        #################################################
        # Refinement
        #################################################

        self.refine = nn.Sequential(

            ConvBNReLU(
                out_channels,
                out_channels),

            ConvBNReLU(
                out_channels,
                out_channels),

            SEBlock(out_channels)
        )

    def forward(self,
                b1,
                b2,
                b3,
                b4):

        #################################################
        # Project Features
        #################################################

        b1 = self.p1(b1)

        b2 = self.p2(b2)

        b3 = self.p3(b3)

        b4 = self.p4(b4)

        #################################################
        # Resize
        #################################################

        target_size = b4.size()[2:]

        b1 = F.interpolate(
            b1,
            size=target_size,
            mode='bilinear',
            align_corners=False)

        b2 = F.interpolate(
            b2,
            size=target_size,
            mode='bilinear',
            align_corners=False)

        b3 = F.interpolate(
            b3,
            size=target_size,
            mode='bilinear',
            align_corners=False)

        #################################################
        # Adaptive Scale Weights
        #################################################

        weights = self.weight_generator(
            [b1, b2, b3, b4])

        w1 = weights[:, 0].view(-1, 1, 1, 1)
        w2 = weights[:, 1].view(-1, 1, 1, 1)
        w3 = weights[:, 2].view(-1, 1, 1, 1)
        w4 = weights[:, 3].view(-1, 1, 1, 1)

        #################################################
        # Dynamic Fusion
        #################################################

        fusion = (
            w1 * b1 +
            w2 * b2 +
            w3 * b3 +
            w4 * b4
        )

        #################################################
        # Refinement
        #################################################

        fusion = self.refine(fusion)

        return fusion






##############################################################
# Boundary Gate
##############################################################

class BoundaryGate(nn.Module):
    """
    Boundary-guided Feature Refinement

    Input:
        Decoder Feature
        Boundary Feature

    Output:
        Refined Decoder Feature
    """

    def __init__(self,
                 decoder_channels,
                 boundary_channels):

        super().__init__()

        self.boundary_proj = nn.Sequential(

            nn.Conv2d(
                boundary_channels,
                decoder_channels,
                kernel_size=1,
                bias=False),

            nn.BatchNorm2d(decoder_channels),

            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(

            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False),

            nn.BatchNorm2d(decoder_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=1),

            nn.Sigmoid()
        )

    def forward(self,
                decoder_feature,
                boundary_feature):

        if boundary_feature.shape[2:] != decoder_feature.shape[2:]:

            boundary_feature = F.interpolate(
                boundary_feature,
                size=decoder_feature.shape[2:],
                mode='bilinear',
                align_corners=False)

        boundary_feature = self.boundary_proj(boundary_feature)

        gate = self.attention(boundary_feature)

        refined = decoder_feature * gate + decoder_feature

        return refined


##############################################################
# Skip Attention Gate
##############################################################

class SkipAttentionGate(nn.Module):

    def __init__(self,
                 encoder_channels,
                 decoder_channels,
                 inter_channels):

        super().__init__()

        self.theta = nn.Sequential(

            nn.Conv2d(
                encoder_channels,
                inter_channels,
                kernel_size=1,
                bias=False),

            nn.BatchNorm2d(inter_channels)
        )

        self.phi = nn.Sequential(

            nn.Conv2d(
                decoder_channels,
                inter_channels,
                kernel_size=1,
                bias=False),

            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(

            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1),

            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                encoder_feature,
                decoder_feature):

        if decoder_feature.shape[2:] != encoder_feature.shape[2:]:

            decoder_feature = F.interpolate(
                decoder_feature,
                size=encoder_feature.shape[2:],
                mode='bilinear',
                align_corners=False)

        theta = self.theta(encoder_feature)

        phi = self.phi(decoder_feature)

        attention = self.relu(theta + phi)

        attention = self.psi(attention)

        encoder_feature = encoder_feature * attention

        return encoder_feature


##############################################################
# Adaptive Skip Fusion
##############################################################

class AdaptiveSkipFusion(nn.Module):

    def __init__(self,
                 encoder_channels,
                 decoder_channels,
                 out_channels):

        super().__init__()

        self.attention = SkipAttentionGate(
            encoder_channels,
            decoder_channels,
            out_channels)

        self.fusion = nn.Sequential(

            ConvBNReLU(
                encoder_channels + decoder_channels,
                out_channels),

            ConvBNReLU(
                out_channels,
                out_channels),

            SEBlock(out_channels)
        )

    def forward(self,
                encoder_feature,
                decoder_feature):

        encoder_feature = self.attention(
            encoder_feature,
            decoder_feature)

        if decoder_feature.shape[2:] != encoder_feature.shape[2:]:

            decoder_feature = F.interpolate(
                decoder_feature,
                size=encoder_feature.shape[2:],
                mode='bilinear',
                align_corners=False)

        fusion = torch.cat(
            [
                encoder_feature,
                decoder_feature
            ],
            dim=1)

        fusion = self.fusion(fusion)

        return fusion


##############################################################
# Boundary Guided Fusion
##############################################################

class BoundaryGuidedFusion(nn.Module):

    def __init__(self,
                 channels,
                 boundary_channels):

        super().__init__()

        self.boundary_gate = BoundaryGate(
            channels,
            boundary_channels)

        self.refine = nn.Sequential(

            ResidualConv(
                channels,
                channels),

            SEBlock(channels)
        )

    def forward(self,
                decoder_feature,
                boundary_feature):

        decoder_feature = self.boundary_gate(
            decoder_feature,
            boundary_feature)

        decoder_feature = self.refine(decoder_feature)

        return decoder_feature




##############################################################
# Adaptive Decoder Block
##############################################################

class AdaptiveDecoderBlock(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            out_channels,
            boundary_channels=256):

        super().__init__()

        ####################################################
        # Upsampling
        ####################################################

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False)

        ####################################################
        # Adaptive Skip Fusion
        ####################################################

        self.skip_fusion = AdaptiveSkipFusion(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels)

        ####################################################
        # Boundary Guided Fusion
        ####################################################

        self.boundary_fusion = BoundaryGuidedFusion(
            out_channels,
            boundary_channels)

        ####################################################
        # Residual Refinement
        ####################################################

        self.refine = nn.Sequential(

            ResidualConv(
                out_channels,
                out_channels),

            ConvBNReLU(
                out_channels,
                out_channels),

            SEBlock(out_channels)
        )

    def forward(
            self,
            decoder_feature,
            encoder_feature,
            boundary_feature):

        ####################################################
        # Upsample
        ####################################################

        decoder_feature = self.up(decoder_feature)

        ####################################################
        # Adaptive Skip Fusion
        ####################################################

        decoder_feature = self.skip_fusion(
            encoder_feature,
            decoder_feature)

        ####################################################
        # Boundary Guidance
        ####################################################

        decoder_feature = self.boundary_fusion(
            decoder_feature,
            boundary_feature)

        ####################################################
        # Final Refinement
        ####################################################

        decoder_feature = self.refine(decoder_feature)

        return decoder_feature





##############################################################
# Decoder Head
##############################################################

class DecoderHead(nn.Module):

    def __init__(self):

        super().__init__()

        ####################################################
        # Decoder Stages
        ####################################################

        self.decoder4 = AdaptiveDecoderBlock(
            encoder_channels=256,
            decoder_channels=512,
            out_channels=256)

        self.decoder3 = AdaptiveDecoderBlock(
            encoder_channels=128,
            decoder_channels=256,
            out_channels=128)

        self.decoder2 = AdaptiveDecoderBlock(
            encoder_channels=64,
            decoder_channels=128,
            out_channels=64)

        self.decoder1 = AdaptiveDecoderBlock(
            encoder_channels=64,
            decoder_channels=64,
            out_channels=64)

    def forward(
            self,
            e1,
            e2,
            e3,
            e4,
            bottleneck,
            boundary_feature):

        d4 = self.decoder4(
            bottleneck,
            e4,
            boundary_feature)

        d3 = self.decoder3(
            d4,
            e3,
            boundary_feature)

        d2 = self.decoder2(
            d3,
            e2,
            boundary_feature)

        d1 = self.decoder1(
            d2,
            e1,
            boundary_feature)

        return d1




##############################################################
# Segmentation Head
##############################################################

class SegmentationHead(nn.Module):

    def __init__(
            self,
            in_channels=64,
            num_classes=6):

        super().__init__()

        self.head = nn.Sequential(

            ConvBNReLU(
                in_channels,
                64),

            nn.Dropout2d(0.1),

            nn.Conv2d(
                64,
                num_classes,
                kernel_size=1)
        )

    def forward(self, x):

        return self.head(x)



##############################################################
# ResNet34 Encoder
##############################################################

class ResNet34Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        ####################################################
        # Boundary Attention for every encoder stage
        ####################################################

        self.bada1 = BoundaryAwareDynamicAttention(64)
        self.bada2 = BoundaryAwareDynamicAttention(64)
        self.bada3 = BoundaryAwareDynamicAttention(128)
        self.bada4 = BoundaryAwareDynamicAttention(256)

        ####################################################
        # Boundary Confidence
        ####################################################

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

        ####################################################
        # Boundary Extraction
        ####################################################

        b1, _ = self.bada1(e1)
        b1, _ = self.bce1(b1)

        b2, _ = self.bada2(e2)
        b2, _ = self.bce2(b2)

        b3, _ = self.bada3(e3)
        b3, _ = self.bce3(b3)

        b4, _ = self.bada4(e4)
        b4, _ = self.bce4(b4)

        return e1, e2, e3, e4, bottleneck, b1, b2, b3, b4




##############################################################
# BADF-Net
##############################################################

class BADF_Net(nn.Module):

    def __init__(self,
                 num_classes=2):

        super().__init__()

        ####################################################
        # Encoder
        ####################################################

        self.encoder = ResNet34Encoder()

        ####################################################
        # Cross-scale Boundary Fusion
        ####################################################

        self.csbf = CrossScaleBoundaryFusion(
            in_channels=(64,64,128,256),
            out_channels=256
        )

        ####################################################
        # Decoder
        ####################################################

        self.decoder = DecoderHead()

        ####################################################
        # Segmentation Head
        ####################################################

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            num_classes=num_classes
        )

    def forward(self, x):

        ####################################################
        # Encoder
        ####################################################

        e1, e2, e3, e4, bottleneck, b1, b2, b3, b4 = self.encoder(x)

        ####################################################
        # Cross-scale Boundary Fusion
        ####################################################

        boundary_feature = self.csbf(
            b1,
            b2,
            b3,
            b4
        )

        ####################################################
        # Decoder
        ####################################################

        decoder_feature = self.decoder(
            e1,
            e2,
            e3,
            e4,
            bottleneck,
            boundary_feature
        )

        ####################################################
        # Segmentation
        ####################################################

        output = self.segmentation_head(
            decoder_feature
        )

        ####################################################
        # Upsample to input resolution
        ####################################################

        output = F.interpolate(
            output,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return output




