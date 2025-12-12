import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
from network.audionet.faeture_extraction import SpecEncoderGlobal, DepthResNet18Encoder,HeatmapDecoder,gaussian_render_batch


class SSLNet(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,
    ):
        super().__init__()

        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
            channels=(16, 32, 64),
            dropout=0.1,
            out_dim=spec_out_dim,
            use_compress=use_compress
        )

        self.depth_encoder = DepthResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )

        for p in self.depth_encoder.features.parameters():
            p.requires_grad = False

        self.fusion_fc = nn.Sequential(
            nn.Linear(spec_out_dim+depth_out_dim, fusion_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.out_dim = fusion_out_dim
        self.decoder = HeatmapDecoder()

        self.film_gamma = nn.Linear(depth_out_dim, spec_out_dim)
        self.film_beta  = nn.Linear(depth_out_dim, spec_out_dim)

    def forward(self, spectrogram, depth):
        spec_feat = self.spec_encoder(spectrogram)  # (B, spec_out_dim)
        # depth_feat = self.depth_encoder(depth)      # (B, depth_out_dim)
        # fused_feat = torch.cat([spec_feat, depth_feat], dim=1)  # (B, spec+depth)
        # out_feat = self.fusion_fc(fused_feat)                    # (B, fusion_out_dim)
        heatmap = self.decoder(spec_feat)                        
        return heatmap
    
class SSLNet_coord(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=256,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,
    ):
        super().__init__()

        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
            channels=(16, 32, 64),
            dropout=0.1,
            out_dim=spec_out_dim,
            use_compress=use_compress
        )

        self.depth_encoder = DepthResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(spec_out_dim + depth_out_dim, fusion_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.out_dim = fusion_out_dim
        self.coord_head = nn.Linear(fusion_out_dim, 2)

    def forward(self, spectrogram, depth):
        spec_feat = self.spec_encoder(spectrogram)  # (B, spec_out_dim)
        # depth_feat = self.depth_encoder(depth)      # (B, depth_out_dim)

        # fused_feat = torch.cat([spec_feat, depth_feat], dim=1)  # (B, spec+depth)
        # out_feat = self.fusion_fc(fused_feat)                    # (B, fusion_out_dim)
        # heatmap = self.decoder(spec_feat)    
        coord = self.coord_head(spec_feat)
        print(coord)
        heatmap = gaussian_render_batch(coord)
        return heatmap


if __name__ == "__main__":
    audio = torch.randn(4, 2, 65, 26)
    depth = torch.randn(4, 1, 128, 128)
    model = SSLNet()
    out = model(audio, depth)
    print(out.shape)  


