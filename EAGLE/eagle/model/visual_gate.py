import torch

def visual_gate(attn_maps, word_spans, last_layers, tau_vis, lam):
    """Compute visuality scores for candidate word units.

    Parameters
    ----------
    attn_maps: ``torch.Tensor``
        Cross-attention maps of shape ``(layers, heads, seq_len, src_len)``.
    word_spans: ``Iterable``
        Iterable of ``(start, end)`` indices specifying token spans for
        candidate word units. ``end`` is exclusive.
    last_layers: ``int``
        Number of final layers whose attention maps are used.
    tau_vis: ``float``
        Visuality threshold. Returned scores are not thresholded, but the
        caller may compare them with ``tau_vis``.
    lam: ``float``
        Scaling factor for the score.

    Returns
    -------
    ``torch.Tensor``
        Tensor of shape ``(len(word_spans),)`` containing visuality scores for
        each span.
    """
    if attn_maps is None or len(attn_maps) == 0:
        return torch.zeros(len(word_spans))

    if isinstance(attn_maps, (list, tuple)):
        attn_maps = torch.stack(attn_maps, dim=0)

    # focus on the last `last_layers` layers
    attn_maps = attn_maps[-last_layers:]

    # average over layers and heads
    attn_maps = attn_maps.mean(dim=0).mean(dim=0)  # (seq_len, src_len)

    scores = []
    for start, end in word_spans:
        if end <= start:
            scores.append(torch.tensor(0.0, device=attn_maps.device))
            continue
        span_attn = attn_maps[start:end]
        score = span_attn.mean()
        scores.append(score)

    scores = torch.stack(scores)
    scores = scores * lam
    return scores
