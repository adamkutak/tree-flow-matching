import torch


def score_si_linear(x, t_batch, u_t):
    """
    x        : (B,C,H,W)
    t_batch  : (B,) scalar times in (0,1]
    u_t      : velocity tensor (same shape as x)
    """
    one_minus_t = 1.0 - t_batch.view(-1, *([1] * (x.ndim - 1)))
    t = t_batch.view(-1, *([1] * (x.ndim - 1)))

    # s(x,t) = -((1-t) * u_t + x) / t
    return -((one_minus_t * u_t + x) / t)


def divfree_swirl_si(x, t_batch, y, v_fn, lambda_div=0.1, eps=1e-8):
    """
    Same signature as before, but the score comes from Eq.(9)
    — no autograd trace needed.
    """
    eps_raw = torch.randn_like(x)
    u_t = v_fn(t_batch, x, y)
    score = score_si_linear(x, t_batch, u_t)

    proj = (eps_raw * score).sum(dim=tuple(range(1, x.ndim)), keepdim=True) / (
        score.norm(dim=tuple(range(1, x.ndim)), keepdim=True) ** 2 + eps
    )
    w = eps_raw - proj * score
    return lambda_div * w
