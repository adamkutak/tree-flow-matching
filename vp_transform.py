# --------------------------------------------------
#  VP helper — analytic, no NaNs, ODE version
# --------------------------------------------------
import torch, math


def vp_tables_ode(n_steps: int, device, beta_min=0.0001, beta_max=0.02, eps=1e-5):
    """
    Build grids (s,t,c,dt,dc) for ODE sampling on a VP path.

    * linear train path :  α=1-t , σ=t
    * VP path (DDPM-linear β) :  β(s)=β_min+(β_max-β_min)s
    """

    # ----- VP schedule (linear-β) --------------------
    s_fwd = torch.linspace(eps, 1.0, n_steps + 1, device=device)  # 0→1
    ds = s_fwd[1] - s_fwd[0]
    beta = beta_min + (beta_max - beta_min) * s_fwd
    beta_int = torch.cumsum(beta, 0) * ds

    bar_alpha = torch.exp(-0.5 * beta_int)
    bar_sigma = torch.sqrt(1.0 - bar_alpha**2)
    bar_rho = bar_alpha / bar_sigma  # ᾱ/σ̄

    # ----- map to linear time ------------------------
    t_fwd = 1.0 / (1.0 + bar_rho)  # analytic inverse
    t_fwd = t_fwd.clamp_min(eps)  # avoid 0
    c_fwd = bar_sigma / t_fwd  # σ̄ / σ(t)

    # ----- analytic derivatives ---------------------
    # bar_alpha' = -0.5 β ᾱ    ,   bar_sigma' = -bar_alpha' bar_alpha / bar_sigma
    bar_alpha_dot = -0.5 * beta * bar_alpha
    bar_sigma_dot = -bar_alpha * bar_alpha_dot / bar_sigma
    bar_rho_dot = (bar_alpha_dot * bar_sigma - bar_alpha * bar_sigma_dot) / bar_sigma**2

    t_dot = -bar_rho_dot / (1.0 + bar_rho) ** 2
    c_dot = (t_fwd * bar_sigma_dot - bar_sigma * t_dot) / (t_fwd**2)

    # replace ill-conditioned last element
    t_dot[-1] = t_dot[-2]
    c_dot[-1] = c_dot[-2]

    # flip to descending order 1→eps
    flip = lambda x: torch.flip(x, [0])
    return tuple(map(flip, (s_fwd, t_fwd, c_fwd, t_dot, c_dot)))
