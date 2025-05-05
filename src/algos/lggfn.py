import torch
from tqdm import tqdm


from gfn.env import Env
from gfn.gflownet import TBGFlowNet



def train_lggfn(
    env: Env,
    mainGFN:TBGFlowNet,
    auxGFN:TBGFlowNet,
    batch_size: int = 16,
    iterations: int = 10,
    lamda: float = 1.0,
    lr: float = 1e-3,
    lr_Z: float = 1e-1,
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


    for _ in tqdm(range(iterations)):
        auxTrajectories = auxGFN.sample_trajectories(
            env=env, n=batch_size, save_logprobs=True
        )
        mainTrajectories = mainGFN.sample_trajectories(
            env=env, n=batch_size // 2, save_logprobs=True
        )
        mainOptimizer.zero_grad()
        auxOptimizer.zero_grad()


        mainAuxLoss = mainGFN.loss(
            env, auxTrajectories, recalculate_all_logprobs=True, reduction="none"
        )
        updated_log_rewards = torch.logsumexp(torch.stack([torch.log(lamda * mainAuxLoss.detach()), auxTrajectories.log_rewards], dim=0), dim=0) #type: ignore

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
