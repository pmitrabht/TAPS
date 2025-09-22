
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftBinnedECE(nn.Module):
    """
    Differentiable Expected Calibration Error via soft binning.

    For each sample:
      - p = softmax(logits)
      - p_true = probability assigned to the ground-truth class (soft "correctness")
      - conf = max probability (or p_true, configurable)
    We softly assign conf to K bins with a Gaussian kernel controlled by temperature T,
    and compute:
      ECE = sum_b w_b * |acc_b - conf_b|
    where acc_b is the soft average of p_true in the bin, conf_b is the soft
    average of conf in the bin, and w_b is the bin mass (normalized).

    Everything is differentiable due to soft assignments.
    """

    def __init__(self, num_bins: int = 15, temperature: float = 0.075,
                 use_max_conf: bool = True, eps: float = 1e-8, p_norm: int = 1):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
        self.use_max_conf = use_max_conf
        self.eps = eps
        self.p_norm = p_norm  # 1 = L1 (|.|), 2 = L2

        # Fixed bin centers in [0,1]
        self.register_buffer(
            "bin_centers",
            torch.linspace(0.0 + 0.5 / num_bins, 1.0 - 0.5 / num_bins, num_bins)
        )

    def set_temperature(self, new_temp: float):
        """Optionally update temperature during training (for decay schedules)."""
        self.temperature = new_temp

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)
        targets: (B,) int64
        """
        B, C = logits.shape
        probs = F.softmax(logits, dim=1)  # (B, C)

        # p_true: probability given to the true class (soft "correctness")
        p_true = probs[torch.arange(B, device=logits.device), targets]  # (B,)

        # confidence: either max prob (classic ECE) or p_true (fully soft)
        if self.use_max_conf:
            conf, _ = probs.max(dim=1)  # (B,)
        else:
            conf = p_true  # avoids argmax non-differentiability

        # Soft assignment to bins with a Gaussian kernel
        conf_exp = conf.unsqueeze(1)             # (B,1)
        centers = self.bin_centers.unsqueeze(0)  # (1,K)
        h2 = (self.temperature ** 2) * 2.0
        weights = torch.exp(- (conf_exp - centers) ** 2 / (h2 + self.eps))  # (B,K)
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)   # (B,K)

        # Bin masses (normalize to sum to 1 across bins)
        w_b = weights.sum(dim=0) + self.eps       # (K,)
        w_b = w_b / w_b.sum()

        # Soft bin-averages of accuracy proxy and confidence
        acc_b = (weights * p_true.unsqueeze(1)).sum(dim=0) / (weights.sum(dim=0) + self.eps)  # (K,)
        conf_b = (weights * conf.unsqueeze(1)).sum(dim=0) / (weights.sum(dim=0) + self.eps)   # (K,)

        # Per-bin gap and aggregated ECE
        if self.p_norm == 1:
            gap_b = (acc_b - conf_b).abs()
        else:
            gap_b = (acc_b - conf_b) ** 2

        ece = (w_b * gap_b).sum()
        return ece
