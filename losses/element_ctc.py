"""CTC loss on the sequence of song elements, trained from clip labels alone.

A Yellowhammer song is a phrase of repeated syllables followed by terminal elements, and the class
of a clip says which elements it contains:

    BC          phrase  B  C          (C: long falling whistle, ~9 -> 6.5 kHz)
    BE          phrase  B  E          (E: short rising whistle, 7-8 kHz)
    BhBl        phrase  B  L          (L: a second whistle, lower than B)
    BlBh        phrase  B  H          (H: a second whistle, higher than B, flat)
    XB          phrase  X  B          (X: short multi-band element, 7-8 kHz)
    Unfinished  phrase | phrase B | phrase X      (no terminal element, or only one)
    None        (nothing)

Connectionist temporal classification (CTC) needs no alignment: it sums over every way of placing
the tokens on the time axis. Where the label allows several sequences (Unfinished, or a weak
candidate set such as `BC|Unfinished|None`) the loss is the negative log of the SUMMED likelihood
of the allowed sequences, so the model itself decides which one a clip is.

Why: a single pooled "is the song complete?" score cannot say WHICH element is missing; "phrase + X
without B" being called XB was the largest remaining confusion. The token sequence makes the
presence of each element an explicit output, and it can be read by a person.


The decoder is generic: nothing below knows about Yellowhammers. A project describes its
vocalisation as an `ElementGrammar` (the element names, and every dialect written as its ending);
`STRNADI_GRAMMAR` is the table of this project. A BioAcu project for another species
needs only its own grammar and clip-level labels.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from config import CLASSES


@dataclass
class ElementGrammar:
    """One table per project: the elements, and every dialect written as its ending.

    Everything else is derived from the table:

    - training targets: a dialect is `song_start` followed by its ending; the incomplete class is
      `song_start` followed by any PROPER PREFIX of a defined ending (for Strnadi: phrase,
      phrase B, phrase X); the nothing class is the empty sequence;
    - decoding (`songs`, `class_of_song`, `class_of_sequence`): the decoded tokens are split into
      songs at every `song_start` token and each song's ending is looked up in the table. An exact
      match is that dialect (so with both `X B` and `X B C` defined, `X B C` is the longer one), a
      proper prefix is the incomplete class, anything else is `OTHER` and is handed to a person
      together with the raw sequence.

    A prefix conflict (is `X B` the dialect XB, or an unfinished XBC?) is visible in the table and
    is the experts' question; the exact match wins here.

    Args:
        elements: names of the sound elements (the CTC blank is added as token 0).
        dialects: {class name: its ending, a list of elements}.
        classes: class names in the order of the model's target vector.
        song_start: the element every song begins with.
        nothing_class / incomplete_class: the classes for "no song" and "song without an ending".
    """

    elements: list
    dialects: dict
    classes: list
    song_start: str = 'phrase'
    nothing_class: str = 'None'
    incomplete_class: str = 'Unfinished'

    def __post_init__(self):
        self.tokens = ['blank'] + list(self.elements)
        self.token_index = {name: index for index, name in enumerate(self.tokens)}
        self.ending_class = {tuple(ending): name for name, ending in self.dialects.items()}
        self.prefixes = {
            tuple(ending[:length]) for ending in self.dialects.values() for length in range(len(ending))
        } - set(self.ending_class)

        self.class_sequences = {self.nothing_class: [[]]}
        self.class_sequences[self.incomplete_class] = [
            [self.song_start] + list(prefix) for prefix in sorted(self.prefixes)
        ]
        for name, ending in self.dialects.items():
            self.class_sequences[name] = [[self.song_start] + list(ending)]

    def allowed_sequences(self, target):
        """Token-index sequences allowed by a target vector (one-hot label or candidate set)."""
        sequences = []
        for class_index, class_name in enumerate(self.classes):
            if target[class_index] >= 0.99:
                for sequence in self.class_sequences[class_name]:
                    sequences.append([self.token_index[token] for token in sequence])
        return sequences

    def songs(self, tokens):
        """Split decoded tokens into songs: a new song starts at every `song_start` token.

        Returns a list of endings (the tokens after each song start). Elements before the first
        song start are the ending of a song whose beginning the window cut off.
        """
        endings, current = [], None
        for token in tokens:
            if token == self.song_start:
                if current is not None:
                    endings.append(current)
                current = []
            elif current is None:
                current = [token]
            else:
                current.append(token)
        if current is not None:
            endings.append(current)
        return endings

    def class_of_song(self, ending):
        """The class of one song, given the elements after its phrase."""
        ending = tuple(ending)
        if ending in self.ending_class:
            return self.ending_class[ending]
        if ending in self.prefixes:
            return self.incomplete_class
        return OTHER

    def class_of_sequence(self, tokens):
        """The class of a window: its first song with a dialect decides.

        Without one: `OTHER` if a song has an ending the table does not know, else the incomplete
        class if there is any song, else the nothing class.
        """
        song_classes = [self.class_of_song(ending) for ending in self.songs(tokens)]
        for name in song_classes:
            if name in self.dialects:
                return name
        if OTHER in song_classes:
            return OTHER
        return self.incomplete_class if song_classes else self.nothing_class


#: Returned for a decoded song whose ending is not in the project's table.
OTHER = 'other'

#: The Yellowhammer table (L / H: a second whistle lower / higher than the B before it).
STRNADI_GRAMMAR = ElementGrammar(
    elements=['phrase', 'B', 'C', 'E', 'L', 'H', 'X'],
    dialects={
        'BC': ['B', 'C'],
        'BE': ['B', 'E'],
        'BhBl': ['B', 'L'],
        'BlBh': ['B', 'H'],
        'XB': ['X', 'B'],
    },
    classes=CLASSES,
)
TOKENS = STRNADI_GRAMMAR.tokens


def element_ctc_loss(token_log_probs, targets, weights=None, grammar=STRNADI_GRAMMAR):
    """Mean negative log-likelihood of the allowed element sequences.

    Args:
        token_log_probs: (B, T, n_tokens) log-probabilities per frame.
        targets: (B, 7) target vectors from `training.data.encode_label`; classes with a value
            of 1 are allowed (soft labels without any such class are skipped).
        weights: optional (B,) loss weight per example.
        grammar: the project's `ElementGrammar`.
    """
    batch_size, n_frames, _ = token_log_probs.shape
    targets = targets.detach().cpu()

    # One row per (example, allowed sequence) pair, all scored in a single CTC call.
    owner, flat_tokens, lengths = [], [], []
    for example in range(batch_size):
        for sequence in grammar.allowed_sequences(targets[example]):
            owner.append(example)
            flat_tokens.extend(sequence)
            lengths.append(len(sequence))
    if not owner:
        return token_log_probs.sum() * 0.0

    device = token_log_probs.device
    owner = torch.tensor(owner, device=device)
    pair_log_probs = token_log_probs[owner].transpose(0, 1).float()  # (T, pairs, n_tokens)
    negative_log_likelihood = F.ctc_loss(
        pair_log_probs,
        torch.tensor(flat_tokens, dtype=torch.long, device=device),
        torch.full((len(owner),), n_frames, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
        blank=0,
        reduction='none',
        zero_infinity=True,
    )  # (pairs,)

    # log of the summed likelihood of each example's allowed sequences
    log_likelihood = torch.full((batch_size,), float('-inf'), device=device)
    log_likelihood = log_likelihood.scatter_reduce(
        0, owner, -negative_log_likelihood, reduce='amax', include_self=True
    )  # the largest term, for a stable log-sum-exp
    summed = torch.zeros(batch_size, device=device).scatter_add(
        0, owner, torch.exp(-negative_log_likelihood - log_likelihood[owner])
    )
    has_target = summed > 0
    per_example = -(log_likelihood[has_target] + torch.log(summed[has_target]))

    if weights is None:
        return per_example.mean()
    weights = weights[has_target]
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-6)


def decode_tokens(token_log_probs, grammar=STRNADI_GRAMMAR):
    """Greedy CTC decoding: (B, T, n_tokens) -> list of token-name lists, and their frame spans.

    Returns:
        (sequences, spans): `sequences[i]` is e.g. ['phrase', 'X', 'B']; `spans[i]` holds
        (token, first_frame, last_frame) for every decoded token, for drawing.
    """
    best = token_log_probs.argmax(-1).cpu().numpy()  # (B, T)
    sequences, spans = [], []
    for frames in best:
        tokens, token_spans, previous = [], [], 0
        for frame, token in enumerate(frames):
            if token != 0 and token != previous:
                tokens.append(grammar.tokens[token])
                token_spans.append([grammar.tokens[token], frame, frame])
            elif token != 0:
                token_spans[-1][2] = frame
            previous = token
        sequences.append(tokens)
        spans.append([tuple(span) for span in token_spans])
    return sequences, spans


def class_of_sequence(tokens, grammar=STRNADI_GRAMMAR):
    """The class a decoded element sequence stands for (`ElementGrammar.class_of_sequence`)."""
    return grammar.class_of_sequence(tokens)


def class_log_likelihoods(token_log_probs, grammar=STRNADI_GRAMMAR):
    """Score every class by the CTC likelihood of its sequences: (B, T, n_tokens) -> (B, n_classes).

    Greedy decoding keeps only the single best token per frame, and drops a faint terminal element
    whenever "blank" is slightly more likely there. Summing the likelihood of each class's allowed
    sequences over all alignments uses the whole distribution instead. (It assumes one song per
    window; `decode_tokens` + `class_of_sequence` handle several.)
    """
    batch_size, n_frames, _ = token_log_probs.shape
    device = token_log_probs.device
    scores = torch.full((batch_size, len(grammar.classes)), float('-inf'), device=device)

    for class_index, class_name in enumerate(grammar.classes):
        for sequence in grammar.class_sequences[class_name]:
            tokens = [grammar.token_index[token] for token in sequence]
            negative_log_likelihood = F.ctc_loss(
                token_log_probs.transpose(0, 1).float(),
                torch.tensor(tokens * batch_size, dtype=torch.long, device=device),
                torch.full((batch_size,), n_frames, dtype=torch.long, device=device),
                torch.full((batch_size,), len(tokens), dtype=torch.long, device=device),
                blank=0,
                reduction='none',
                zero_infinity=True,
            )
            scores[:, class_index] = torch.logaddexp(scores[:, class_index], -negative_log_likelihood)
    return scores.log_softmax(1)
