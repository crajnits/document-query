from typing import Tuple

import numpy as np
from transformers.utils import is_pytesseract_available, is_vision_available


VISION_LOADED = False
if is_vision_available():
    from PIL import Image
    from transformers.image_utils import load_image

    VISION_LOADED = True
else:
    Image = None
    load_image = None


TESSERACT_LOADED = False
if is_pytesseract_available():
    import pytesseract

    TESSERACT_LOADED = True
else:
    pytesseract = None


def decode_spans(
    start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int, undesired_tokens: np.ndarray
) -> Tuple:
    """
    Take the output of any `ModelForQuestionAnswering` and will generate probabilities for each span to be the actual
    answer.
    """
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    starts, ends = np.unravel_index(idx_sort, candidates.shape)[1:]
    desired_spans = np.isin(starts, undesired_tokens.nonzero()) & np.isin(ends, undesired_tokens.nonzero())
    starts = starts[desired_spans]
    ends = ends[desired_spans]
    scores = candidates[0, starts, ends]

    return starts, ends, scores


def select_starts_ends(
    start,
    end,
    p_mask,
    attention_mask,
    min_null_score=1000000,
    top_k=1,
    handle_impossible_answer=False,
    max_answer_len=15,
):
    """
    Takes the raw output of any `ModelForQuestionAnswering` and first normalizes its outputs and then uses
    `decode_spans()` to generate probabilities for each span to be the actual answer.
    """
    # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
    undesired_tokens = np.abs(np.array(p_mask) - 1)

    if attention_mask is not None:
        undesired_tokens = undesired_tokens & attention_mask

    # Generate mask
    undesired_tokens_mask = undesired_tokens == 0.0

    # Make sure non-context indexes in the tensor cannot contribute to the softmax
    start = np.where(undesired_tokens_mask, -10000.0, start)
    end = np.where(undesired_tokens_mask, -10000.0, end)

    # Normalize logits and spans to retrieve the answer
    start = np.exp(start - start.max(axis=-1, keepdims=True))
    start = start / start.sum()

    end = np.exp(end - end.max(axis=-1, keepdims=True))
    end = end / end.sum()

    if handle_impossible_answer:
        min_null_score = min(min_null_score, (start[0, 0] * end[0, 0]).item())

    # Mask CLS
    start[0, 0] = end[0, 0] = 0.0

    starts, ends, scores = decode_spans(start, end, top_k, max_answer_len, undesired_tokens)
    return starts, ends, scores, min_null_score
