import torch
from torch import nn
from torch.nn import functional as F
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from torch_geometric.nn import GATConv


class FocusFilterBlock(nn.Module):
    """Pointwise transformation, depthwise filtering, and pointwise projection."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        if hidden_channels != out_channels:
            raise ValueError("The research implementation uses identical hidden and output dimensions.")
        self.expand = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2,
            padding=1, groups=out_channels,
        )
        self.project = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.expand(x))
        x = self.depthwise_conv(x)
        x = self.project(x)
        return F.relu(self.bn(x))


class FullyConnectedGraphAttention(nn.Module):
    """Four-head GAT applied independently to each fully connected region graph."""

    def __init__(self, channels, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.channels = channels
        self.gat = GATConv(
            channels, channels, heads=heads, concat=True,
            dropout=dropout, add_self_loops=False,
        )

    @staticmethod
    def complete_edge_index(batch_size, node_count, device):
        nodes = torch.arange(node_count, device=device)
        source = nodes.repeat_interleave(node_count)
        target = nodes.repeat(node_count)
        base = torch.stack([source, target], dim=0)
        offsets = torch.arange(batch_size, device=device).view(-1, 1, 1) * node_count
        return (base.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

    def forward(self, nodes):
        batch, count, channels = nodes.shape
        edge_index = self.complete_edge_index(batch, count, nodes.device)
        output = F.elu(self.gat(nodes.reshape(batch * count, channels), edge_index))
        output = output.view(batch, count, self.heads, self.channels)
        return output.mean(dim=2)


class SwinImageBranch(nn.Module):
    def __init__(self, feature_dim=96, hidden_dim=96, pretrained=True):
        super().__init__()
        self.backbone = swin_tiny_patch4_window7_224(pretrained=pretrained)
        self.reduce = nn.Conv2d(768, feature_dim, kernel_size=1)
        self.refine = FocusFilterBlock(feature_dim, hidden_dim, feature_dim)
        self.project = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, image):
        feature = self.backbone.forward_features(image).permute(0, 3, 1, 2).contiguous()
        feature = self.reduce(feature)
        feature = self.refine(feature)
        return self.project(feature)


class GazeBranch(nn.Module):
    def __init__(self, feature_dim=96, ffb_hidden_dim=96, conv_hidden_dim=96):
        super().__init__()
        self.refine = FocusFilterBlock(1, ffb_hidden_dim, feature_dim)
        self.conv1 = nn.Conv2d(1, conv_hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_hidden_dim, feature_dim, kernel_size=3, padding=1)

    def forward(self, gaze, target_size):
        refined = self.refine(gaze)
        refined = refined.mean(dim=1, keepdim=True)
        refined = self.conv1(refined)
        refined = torch.sigmoid(self.conv2(refined))
        return F.interpolate(refined, size=target_size, mode="bilinear", align_corners=False)


class SwiGAT(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_dim=96,
        ffb_hidden_dim=96,
        gaze_hidden_dim=96,
        gat_heads=4,
        dropout=0.0,
        pretrained_backbone=True,
    ):
        super().__init__()
        self.image_branch = SwinImageBranch(feature_dim, ffb_hidden_dim, pretrained_backbone)
        self.gaze_branch = GazeBranch(feature_dim, ffb_hidden_dim, gaze_hidden_dim)
        self.gat = FullyConnectedGraphAttention(feature_dim, gat_heads, dropout)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes),
        )

    def forward_features(self, image, gaze):
        refined_xray_map = self.image_branch(image)
        refined_gaze_map = self.gaze_branch(gaze, refined_xray_map.shape[-2:])
        gaze_modulated = refined_xray_map * refined_gaze_map
        batch, channels, height, width = gaze_modulated.shape
        nodes = gaze_modulated.flatten(2).transpose(1, 2)
        nodes = self.gat(nodes)
        graph_refined_map = nodes.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        return graph_refined_map, refined_gaze_map

    def forward(self, image, gaze):
        graph_refined_map, _ = self.forward_features(image, gaze)
        return self.classifier(graph_refined_map)

