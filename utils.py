import torch


def score_si_linear(x, t_batch, u_t):
    t_scalar = t_batch[0].item()

    if t_scalar == 0.0:
        # limit t → 0  gives score = −x
        return -x
    else:
        one_minus_t = 1.0 - t_batch.view(-1, *([1] * (x.ndim - 1)))
        t = t_batch.view(-1, *([1] * (x.ndim - 1)))
        return -((one_minus_t * u_t + x) / t)


def divfree_swirl_si(x, t_batch, y, u_t, eps=1e-8):
    eps_raw = torch.randn_like(x)
    score = score_si_linear(x, t_batch, u_t)

    dims = tuple(range(1, x.ndim))
    dot = (eps_raw * score).sum(dim=dims, keepdim=True)
    s_norm = torch.linalg.vector_norm(score, dim=dims, keepdim=True) + eps
    s_norm2 = s_norm.pow(2)
    proj = dot / s_norm2
    w = eps_raw - proj * score

    return w


def score_vp_converted(x, t_batch, u_t, use_vp=True, beta_min=0.1, beta_max=20.0):
    """
    Convert velocity to score using either linear or VP scheduler coefficients
    Assumes t_batch is already in [0,1] range
    """
    t_normalized = t_batch  # Already in [0,1] - no normalization needed

    if use_vp:
        # VP scheduler coefficients (like repository)
        b = beta_min
        B = beta_max
        T = 0.5 * t_normalized**2 * (B - b) + t_normalized * b
        dT = t_normalized * (B - b) + b

        alpha_t = torch.exp(-0.5 * T)
        sigma_t = torch.sqrt(1 - torch.exp(-T))
        d_alpha_t = -0.5 * dT * torch.exp(-0.5 * T)
        d_sigma_t = 0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T))
    else:
        # Linear interpolant coefficients
        alpha_t = 1 - t_normalized
        sigma_t = t_normalized
        d_alpha_t = -torch.ones_like(t_normalized)
        d_sigma_t = torch.ones_like(t_normalized)

    # Repository's general formula
    alpha_t = alpha_t.view(-1, *([1] * (x.ndim - 1)))
    sigma_t = sigma_t.view(-1, *([1] * (x.ndim - 1)))
    d_alpha_t = d_alpha_t.view(-1, *([1] * (x.ndim - 1)))
    d_sigma_t = d_sigma_t.view(-1, *([1] * (x.ndim - 1)))

    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * u_t - x) / var

    return score
