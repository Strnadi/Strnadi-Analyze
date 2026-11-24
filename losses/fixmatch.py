"""FixMatch consistency (Sohn et al. 2020) with FreeMatch self-adaptive, class-wise thresholds
(Wang et al. 2023).

The EMA teacher labels a weakly augmented view; the student must reproduce that label on a
strongly augmented view, but only where the teacher is confident relative to a threshold that
adapts to how confident the model currently is overall (global) and per class (so rare classes are
not starved).

Caveat from the BirdCLEF write-ups cited in the reviews: pseudo-labels help only when the teacher
is strong; a weak self-teacher made results worse. Prefer a fine-tuned/distilled teacher over the
model's own EMA.

Origin: research session (R4).
"""

import torch
import torch.nn.functional as F


class FreeMatch:
    """Pseudo-labelling with thresholds that follow the teacher's confidence during training.

    Attributes:
        tau: scalar tensor, running average of the teacher's confidence (its highest class
            probability) over all examples seen: the global threshold.
        pc: (n_classes,) tensor, running average of the teacher's class probabilities; classes
            predicted less often get a lower threshold.
        tau_min: lower bound for the global threshold.
        m: momentum of both running averages (close to 1 = slow).
    """

    def __init__(self, n_classes, tau_min=0.5, momentum=0.999):
        """Start both running averages at the uniform value 1 / n_classes.

        Args:
            n_classes: number of classes C.
            tau_min: lower bound for the global confidence threshold.
            momentum: momentum of the running averages.
        """
        uniform_probability = 1.0 / n_classes
        self.tau = torch.tensor(uniform_probability)
        self.pc = torch.full((n_classes,), uniform_probability)
        self.tau_min = tau_min
        self.m = momentum

    def state(self):
        """Return the running averages, e.g. for a checkpoint.

        Returns:
            The tuple (tau, pc).
        """
        return (self.tau, self.pc)

    def load(self, state):
        """Restore the running averages.

        Args:
            state: the tuple (tau, pc) as returned by `state()`.

        Returns:
            Nothing.
        """
        self.tau, self.pc = state

    @torch.no_grad()
    def pseudo_labels(self, teacher_logits):
        """Turn teacher predictions into pseudo-labels and decide which of them to trust.

        Also updates the running averages `tau` and `pc` with this batch.

        Args:
            teacher_logits: (B, C) logits of the teacher on the weakly augmented view.

        Returns:
            (label, mask): `label` (B,) is the teacher's most probable class; `mask` (B,) is 1.0
            where the teacher's confidence reaches the threshold of that class and 0.0 elsewhere.
        """
        teacher_probs = torch.softmax(teacher_logits, 1)  # (B, C)
        confidence, label = teacher_probs.max(1)  # both (B,)
        # the running averages follow the model to its device (they start on the CPU)
        self.tau, self.pc = self.tau.to(teacher_probs.device), self.pc.to(teacher_probs.device)

        # Exponential moving averages of the mean confidence and of the mean class probabilities.
        keep = self.m
        update = 1 - self.m
        self.tau = keep * self.tau + update * confidence.mean()
        self.pc = keep * self.pc + update * teacher_probs.mean(0)

        # Threshold per class: the global threshold (never below tau_min), scaled down for
        # classes the teacher predicts less often than its most frequent class.
        relative_class_frequency = self.pc / self.pc.max()  # (C,), 1 for the top class
        global_threshold = self.tau.clamp(min=self.tau_min)
        class_threshold = relative_class_frequency * global_threshold  # (C,)

        is_confident = confidence >= class_threshold[label]  # (B,)
        return label, is_confident.float()

    def loss(self, student_logits, label, mask):
        """Cross-entropy of the student against the trusted pseudo-labels.

        Args:
            student_logits: (B, C) logits of the student on the strongly augmented view.
            label: (B,) pseudo-labels from `pseudo_labels`.
            mask: (B,) 1.0 for trusted pseudo-labels, 0.0 for the rest.

        Returns:
            Scalar: the masked per-example loss averaged over ALL B examples, so a batch with few
            trusted labels gives a small loss.
        """
        per_example = F.cross_entropy(student_logits, label, reduction='none')  # (B,)
        return (per_example * mask).mean()
