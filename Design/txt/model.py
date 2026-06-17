# ============================================================
# ExpoEscom - Arquitectura del modelo
# MobileNetV2 + cabeza multi-etiqueta
# ============================================================
import torch
import torch.nn as nn
import torchvision.models as models


class CartoonClassifier(nn.Module):
    """
    MobileNetV2 pre-entrenado + cabeza propia.
    num_classes es dinámico según las clases detectadas.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights  = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features        # [B, 1280, 7, 7]
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

        # Cabeza: Linear → ReLU → Dropout → Linear (SIN sigmoid,
        # BCEWithLogitsLoss la aplica internamente).
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # Fase 1: backbone congelado, solo entrena la cabeza
        self._unfrozen = 0
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(torch.flatten(x, 1))    # logits [B, num_classes]

    def train(self, mode=True):
        """
        Mantiene en eval los BatchNorm de los bloques congelados: sus convs
        no se actualizan, así que dejar que sus running stats deriven hacia
        el dataset de caricaturas solo desajusta las features de ImageNet.
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
