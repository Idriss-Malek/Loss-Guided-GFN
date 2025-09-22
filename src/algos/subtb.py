import torch
from tqdm import tqdm
import wandb

from gfn.env import Env
from gfn.gflownet import SubTBGFlowNet  # or wherever SubTBGFlowNet is defined


def train_subtb(
    env: Env,
    mainGFN: SubTBGFlowNet,
    batch_size: int = 16,
    iterations: int = 10,
    lr: float = 1e-3,
    lr_logF: float | None = None,   # optional separate LR for logF head
    device_str: str = "cpu",
    recalc_logprobs: bool = False,  # match your TB loop's behavior
):
    """
    Train a Sub-Trajectory Balance GFlowNet.

    Args:
        env: The environment to train on.
        subGFN: The SubTBGFlowNet model.
        batch_size: Batch size for trajectory sampling.
        iterations: Number of optimization steps.
        lr: Base learning rate for model parameters (excluding logF if lr_logF is set).
        lr_logF: Optional learning rate for logF parameters (if None, uses lr).
        device_str: Device string, e.g., "cpu" or "cuda".
        recalc_logprobs: Whether to recompute log-probs in loss (usually False for speed).

    Returns:
        loss_history: list of float
    """
    device = torch.device(device_str)
    mainGFN.to(device)

    # --- Build optimizer param groups
    # Grab logF params (if present) so we can optionally give them a different LR.
    try:
        logF_params = list(mainGFN.logF_parameters())
    except Exception:
        logF_params = []

    # Everything except logF (avoid duplicates)
    all_named = dict(mainGFN.named_parameters())
    logF_param_ids = {id(p) for p in logF_params}
    base_params = [p for p in all_named.values() if id(p) not in logF_param_ids]

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": lr})
    if logF_params:
        param_groups.append({"params": logF_params, "lr": (lr if lr_logF is None else lr_logF)})

    if not param_groups:
        raise ValueError("No trainable parameters found in subGFN.")

    optimizer = torch.optim.Adam(param_groups)

    loss_history = []

    for step in tqdm(range(iterations)):
        # Sample trajectories with log-probs for STB
        trajectories = mainGFN.sample_trajectories(
            env=env, n=batch_size, save_logprobs=True
        )

        optimizer.zero_grad()

        loss = mainGFN.loss(
            env=env,
            trajectories=trajectories,
            recalculate_all_logprobs=recalc_logprobs,
        )
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        # Minimal logging (mirrors your TB loop)
        # wandb.log({
        #     "subtb_loss": loss_val,
        #     "iteration": step,
        #     "weighting": getattr(mainGFN, "weighting", "unknown"),
        #     "lambda": getattr(mainGFN, "lamda", float("nan")),
        #     "forward_looking": getattr(mainGFN, "forward_looking", False),
        # })

    return loss_history
