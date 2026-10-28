"""Selectable pretrained video backbones for MELD face-frame clips.
Input frames: [B,T,3,H,W], normalized like ImageNet. Output 512-D embedding/logits.
Video transformers are optional and require compatible Transformers/torchvision versions.
"""
import torch
import torch.nn as nn
import torchvision.models as tv

VIDEO_BACKBONES=('mobilenet','videomae','timesformer','swin')

class TemporalAttention(nn.Module):
    def __init__(self,dim): super().__init__(); self.score=nn.Linear(dim,1)
    def forward(self,x):
        w=torch.softmax(self.score(x).squeeze(-1),dim=1)
        return torch.bmm(w.unsqueeze(1),x).squeeze(1)

class AdvancedVideoEmotionModel(nn.Module):
    def __init__(self,backbone='mobilenet',num_classes=7,freeze_backbone=True,frames=4,image_size=224):
        super().__init__(); self.backbone_name=backbone; self.frames=frames
        if backbone not in VIDEO_BACKBONES: raise ValueError(f'backbone must be one of {VIDEO_BACKBONES}')
        if backbone=='mobilenet':
            net=tv.mobilenet_v2(weights=tv.MobileNet_V2_Weights.IMAGENET1K_V2)
            self.frame_encoder=net.features; self.pool2d=nn.AdaptiveAvgPool2d(1); raw_dim=1280
            self.temporal=nn.GRU(raw_dim,256,batch_first=True,bidirectional=True)
            self.temporal_pool=TemporalAttention(512); self.video_encoder=None
        elif backbone in ('videomae','timesformer'):
            try:
                if backbone=='videomae':
                    from transformers import VideoMAEModel
                    self.video_encoder=VideoMAEModel.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
                else:
                    from transformers import TimesformerModel
                    self.video_encoder=TimesformerModel.from_pretrained('facebook/timesformer-base-finetuned-k400')
            except Exception as e:
                raise RuntimeError(f'Could not load pretrained {backbone}. Check transformers version/network/cache: {e}') from e
            raw_dim=int(self.video_encoder.config.hidden_size); self.frame_encoder=None; self.temporal=None; self.temporal_pool=None
        else:
            # Torchvision video Swin is memory-intensive; use its pretrained Kinetics-400 trunk.
            try:
                from torchvision.models.video import swin3d_t, Swin3D_T_Weights
                net=swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
                raw_dim=768
                net.head=nn.Identity()
                self.video_encoder=net; self.frame_encoder=None; self.temporal=None; self.temporal_pool=None
            except Exception as e:
                raise RuntimeError(f'This torchvision build does not expose pretrained Swin3D: {e}') from e
        if freeze_backbone:
            trunk=self.frame_encoder if backbone=='mobilenet' else self.video_encoder
            for p in trunk.parameters(): p.requires_grad=False
        self.projection=nn.Sequential(nn.LayerNorm(raw_dim),nn.Linear(raw_dim,512),nn.GELU(),nn.Dropout(.2))
        self.classifier=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,num_classes))
    def forward(self,frames,return_embedding=False):
        # frames [B,T,C,H,W]
        b,t,c,h,w=frames.shape
        if self.backbone_name=='mobilenet':
            z=self.frame_encoder(frames.reshape(b*t,c,h,w)); z=self.pool2d(z).flatten(1).reshape(b,t,-1)
            z,_=self.temporal(z); pooled=self.temporal_pool(z)
        elif self.backbone_name=='swin':
            z=self.video_encoder(frames.permute(0,2,1,3,4))
            if z.ndim>2: z=z.flatten(2).mean(-1)
            pooled=z
        else:
            z=self.video_encoder(pixel_values=frames).last_hidden_state
            pooled=z[:,0] if z.ndim==3 else z.mean(dim=1)
        embedding=self.projection(pooled); logits=self.classifier(embedding)
        return (logits,embedding) if return_embedding else logits
