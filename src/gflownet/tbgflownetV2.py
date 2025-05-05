from typing_extensions import override
import torch
from typing import Optional, Tuple
import math

from gfn.gflownet import TBGFlowNet
from gfn.env import Env
from gfn.containers import Trajectories
from gfn.utils.handlers import is_callable_exception_handler


class TBGFlowNetV2(TBGFlowNet):

    def get_trajectories_scores(
        self,
        trajectories: Trajectories,
        log_rewards: Optional[torch.Tensor] = None,
        recalculate_all_logprobs: bool = False,
        with_pb: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a batch of trajectories, calculate forward & backward policy scores.

        Args:
            trajectories: Trajectories to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns: A tuple of float tensors of shape (n_trajectories,)
            containing the total log_pf, total log_pb, and the total
            log-likelihood of the trajectories.
        """
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        assert log_pf_trajectories is not None
        total_log_pf_trajectories = log_pf_trajectories.sum(dim=0)
        total_log_pb_trajectories = log_pb_trajectories.sum(dim=0)

        if log_rewards is None:
            log_rewards = trajectories.log_rewards

        if math.isfinite(self.log_reward_clip_min) and log_rewards is not None:
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)

        if torch.any(torch.isinf(total_log_pf_trajectories)) or torch.any(
            torch.isinf(total_log_pb_trajectories)
        ):
            raise ValueError("Infinite logprobs found")

        assert total_log_pf_trajectories.shape == (trajectories.n_trajectories,)
        assert total_log_pb_trajectories.shape == (trajectories.n_trajectories,)
        if with_pb:
            return (
                total_log_pf_trajectories,
                total_log_pb_trajectories,
                total_log_pf_trajectories - total_log_pb_trajectories - log_rewards, 
            )
        else:
            return (
                total_log_pf_trajectories,
                total_log_pb_trajectories,
                total_log_pf_trajectories - log_rewards,
            )

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        log_rewards: Optional[torch.Tensor] = None,
        recalculate_all_logprobs: bool = False,
        reduction:str = 'mean'
    ) -> torch.Tensor:
        """Trajectory balance loss.

        The trajectory balance loss is described in 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259))

        Raises:
            ValueError: if the loss is NaN.
        """
        del env  # unused
        _, _, scores = self.get_trajectories_scores(
            trajectories,
            log_rewards=log_rewards,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )

        # If the conditioning values exist, we pass them to self.logZ
        # (should be a ScalarEstimator or equivilant).
        if trajectories.conditioning is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                logZ = self.logZ(trajectories.conditioning)
        else:
            logZ = self.logZ

        if reduction == 'mean':
            loss = (scores + logZ.squeeze()).pow(2).mean()
        elif reduction == 'sum':
            loss = (scores + logZ.squeeze()).pow(2).sum()
        elif reduction == 'none':
            loss = (scores + logZ.squeeze()).pow(2)
        else:
            raise ValueError(f"Unknown reduce option: {reduction}")
        if torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss