import torch
from tqdm import tqdm
import wandb

from gfn.env import Env
from gfn.gflownet import TBGFlowNet


def train_tb(
    env: Env,
    mainGFN:TBGFlowNet,
    batch_size: int = 16,
    iterations: int = 10,
    lr: float = 1e-3,
    lr_Z: float = 1e-1,
    device_str: str = "cpu",
):
    """
    Train a TB GFlowNet.

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

    preprocessor = mainGFN.pf.preprocessor

    main_non_logz_params = [
        v for k, v in dict(mainGFN.named_parameters()).items() if k != "logZ"
    ]

    mainOptimizer = torch.optim.Adam(main_non_logz_params, lr=lr)
    logz_params = [dict(mainGFN.named_parameters())["logZ"]]
    mainOptimizer.add_param_group({"params": logz_params, "lr": lr_Z})

    loss_history = []

    for step in tqdm(range(iterations)):

        mainTrajectories = mainGFN.sample_trajectories(
            env=env, n=batch_size, save_logprobs=True
        )
        mainOptimizer.zero_grad()

        mainLoss = mainGFN.loss(
            env, mainTrajectories, recalculate_all_logprobs=False
        )
        mainLoss.backward()
        mainOptimizer.step()

        loss_history.append(mainLoss.item())

        wandb.log({
        "main_loss": mainLoss.item(),
        "iteration": step
        })
        

    return loss_history


    


        



