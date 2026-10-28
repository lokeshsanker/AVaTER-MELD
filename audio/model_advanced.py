"""Sequence-aware Wav2Vec2/HuBERT audio encoder for MELD.
Output: 768-D embedding plus 7-class logits. Intended for raw 16-kHz waveforms.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

BACKBONES={
 'wav2vec2':'facebook/wav2vec2-base-960h',
 'hubert':'facebook/hubert-base-ls960',
}

class AudioSequenceEmotionModel(nn.Module):
    def __init__(self, backbone='wav2vec2', num_classes=7, lstm_hidden=256, freeze_encoder=True, pretrained_name=None):
        super().__init__()
        if backbone not in BACKBONES: raise ValueError(f'backbone must be one of {tuple(BACKBONES)}')
        self.backbone_name=backbone
        self.encoder=AutoModel.from_pretrained(pretrained_name or BACKBONES[backbone])
        dim=int(self.encoder.config.hidden_size)
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad=False
        self.bilstm=nn.LSTM(dim,lstm_hidden,batch_first=True,bidirectional=True,num_layers=1)
        self.attention=nn.Sequential(nn.Linear(2*lstm_hidden,128),nn.Tanh(),nn.Linear(128,1))
        self.projection=nn.Sequential(nn.LayerNorm(2*lstm_hidden),nn.Linear(2*lstm_hidden,768),nn.GELU(),nn.Dropout(.2))
        self.classifier=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,num_classes))
    def forward(self,input_values,attention_mask=None,return_embedding=False):
        hidden=self.encoder(input_values=input_values,attention_mask=attention_mask).last_hidden_state
        seq,_=self.bilstm(hidden)
        scores=self.attention(seq).squeeze(-1)
        weights=torch.softmax(scores,dim=1)
        pooled=torch.bmm(weights.unsqueeze(1),seq).squeeze(1)
        embedding=self.projection(pooled)
        logits=self.classifier(embedding)
        return (logits,embedding) if return_embedding else logits
