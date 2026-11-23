"""FixMatch consistency (Sohn et al. 2020) with FreeMatch self-adaptive, class-wise thresholds (Wang et al. 2023).

The EMA teacher labels a weakly augmented view; the student must reproduce that label on a strongly augmented
view, but only where the teacher is confident relative to a threshold that adapts to how confident the model
currently is overall (global) and per class (so rare classes are not starved).

Caveat from the BirdCLEF write-ups cited in the reviews: pseudo-labels help only when the teacher is strong;
a weak self-teacher made results worse. Prefer a fine-tuned/distilled teacher over the model's own EMA.
Origin: research session (R4).
"""
import torch
import torch.nn.functional as F


class FreeMatch:
    def __init__(self, n_classes, tau_min=0.5, momentum=0.999):
        self.tau = torch.tensor(1.0 / n_classes); self.pc = torch.full((n_classes,), 1.0 / n_classes)
        self.tau_min, self.m = tau_min, momentum

    def state(self):
        return (self.tau, self.pc)

    def load(self, state):
        self.tau, self.pc = state

    @torch.no_grad()
    def pseudo_labels(self, teacher_logits):
        p = torch.softmax(teacher_logits, 1); conf, label = p.max(1)
        self.tau = self.m * self.tau + (1 - self.m) * conf.mean()
        self.pc = self.m * self.pc + (1 - self.m) * p.mean(0)
        thr = (self.pc / self.pc.max()) * self.tau.clamp(min=self.tau_min)
        return label, (conf >= thr[label]).float()

    def loss(self, student_logits, label, mask):
        return (F.cross_entropy(student_logits, label, reduction='none') * mask).mean()
