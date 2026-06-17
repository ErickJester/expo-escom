# ============================================================
# ExpoEscom - Arquitectura del modelo (LOCAL v3.0)
# MobileNetV2 + cabeza single-label (CrossEntropyLoss)
# ============================================================
import torch
import torch.nn as nn
import torchvision.models as models


class CartoonClassifier(nn.Module):
    """
    MobileNetV2 pre-entrenado + cabeza propia.
    - extract(x): vector de features [B, 1280] (para el feature caching).
    - forward(x): logits [B, num_classes].
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights  = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features          # [B, 1280, 7, 7]
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            # SIN Softmax → CrossEntropyLoss lo aplica internamente
        )

        self._unfrozen = 0
        for p in self.features.parameters():
            p.requires_grad = False

    def extract(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)                 # [B, 1280]

    def forward(self, x):
        return self.head(self.extract(x))          # logits

    def train(self, mode=True):
        """
        Mantiene en eval los BatchNorm de los bloques congelados: sus convs
        no se actualizan, así que dejar derivar sus running stats solo
        desajusta las features de ImageNet.
        """
        super().train(mode)
        if mode:
            blocks = list(self.features.children())
            frozen = blocks[:len(blocks) - self._unfrozen]
            for block in frozen:
                for m in block.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        return self

    def unfreeze_last_n(self, n=3):
        """Descongela los últimos n bloques del backbone (Fase 2)."""
        self._unfrozen = n
        for layer in list(self.features.children())[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   🔥 {n} bloques descongelados → {t:,} params entrenables")
