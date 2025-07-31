import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from gfn.env import Env
from gfn.utils.modules import MLP
from gfn.gflownet import TBGFlowNet


def train_sagfn(
    env: Env,
    mainGFN:TBGFlowNet,
    auxGFN:TBGFlowNet,
    batch_size: int = 16,
    iterations: int = 10,
    reward_scale: float = 1.0,
    beta_e: float = 1.0,
    beta_i: float = 1.0,
    beta_sn: float = 1.0,
    lr: float = 1e-3,
    lr_Z: float = 1e-1,
    lr_rnd: float = 1e-3,
    device_str: str = "cpu",
):
    """
    Train a siblings augmented GFlowNet.

    Args:
        env: The environment to train on.
        mainGFN (TBGFlowNetV2): The main GFN model.
        auxGFN (TBGFlowNetV2): The auxiliary GFN model.
        batch_size (int): The size of the training batches.
        iterations (int): The number of training epochs.

    Returns:
        None
    """

    device = torch.device(device_str)
    mainGFN.to(device)
    auxGFN.to(device)

    preprocessor = mainGFN.pf.preprocessor

    main_non_logz_params = [
        v for k, v in dict(mainGFN.named_parameters()).items() if k != "logZ"
    ]
    mainOptimizer = torch.optim.Adam(main_non_logz_params, lr=lr)
    logz_params = [dict(mainGFN.named_parameters())["logZ"]]
    mainOptimizer.add_param_group({"params": logz_params, "lr": lr_Z})

    aux_non_logz_params = [
        v for k, v in dict(auxGFN.named_parameters()).items() if k != "logZ"
    ]
    auxOptimizer = torch.optim.Adam(aux_non_logz_params, lr=lr)
    auxlogz_params = [dict(auxGFN.named_parameters())["logZ"]]
    auxOptimizer.add_param_group({"params": auxlogz_params, "lr": lr_Z})

    rnd_fix = MLP(input_dim=preprocessor.output_dim, output_dim=1)
    rnd_train = MLP(input_dim=preprocessor.output_dim, output_dim=1)
    rnd_fix.to(device)
    rnd_train.to(device)
    for param in rnd_fix.parameters():
        param.requires_grad = False

    rndOptimizer = torch.optim.Adam(rnd_train.parameters(), lr=lr_rnd)

    main_loss_history = []
    aux_loss_history = []


    for step in tqdm(range(iterations)):
        
        auxTrajectories = auxGFN.sample_trajectories(
            env=env, n=batch_size, save_logprobs=True
        )
        mainTrajectories = mainGFN.sample_trajectories(
            env=env, n=batch_size // 2, save_logprobs=True
        )
        mainOptimizer.zero_grad()
        auxOptimizer.zero_grad()
        rndOptimizer.zero_grad()

        auxLastStates = preprocessor(auxTrajectories.terminating_states)
        rnd_loss = reward_scale * F.mse_loss(
            rnd_fix(auxLastStates), rnd_train(auxLastStates), reduction="none"
        ).squeeze(-1)
        mainAuxLoss = mainGFN.loss(
            env, auxTrajectories, recalculate_all_logprobs=True, reduction="none"
        )
        a = beta_e * auxTrajectories.log_rewards #type: ignore
        b = beta_i * rnd_loss.log().detach()
        updated_log_rewards = beta_sn * torch.logsumexp(torch.stack([a, b], dim=0), dim=0)

        mainMainLoss = mainGFN.loss(
            env, mainTrajectories, recalculate_all_logprobs=False, reduction="none"
        )
        mainLoss = torch.cat([mainAuxLoss[: batch_size//2], mainMainLoss]).mean()
        mainLoss.backward()
        mainOptimizer.step()
        auxLoss = auxGFN.loss(
            env,
            auxTrajectories,
            log_rewards=updated_log_rewards,
            recalculate_all_logprobs=False,
        )
        auxLoss.backward()
        auxOptimizer.step()
        rnd_loss.mean().backward()
        rndOptimizer.step()

        main_loss_history.append(mainLoss.item())
        aux_loss_history.append(auxLoss.item())

        wandb.log({
        "main_loss": mainLoss.item(),
        "aux_loss": auxLoss.item(),
        "iteration": step
        })

    return main_loss_history, aux_loss_history
