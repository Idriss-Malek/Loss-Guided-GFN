import torch
import math
from typing import Optional
from gfn.preprocessors import Preprocessor

from gfn.gym.bitSequence import BitSequenceStates, BitSequence 

class BalancedParentheses(BitSequence):
    def __init__(
        self,
        word_size: int = 4,
        seq_size: int = 120,
        device_str: str = 'cpu',
    ):
        assert seq_size % word_size == 0, f"word_size {word_size} must divide seq_size {seq_size}."
        self.words_per_seq : int = seq_size // word_size
        self.word_size : int = word_size
        self.seq_size : int = seq_size
        n_actions = 2**word_size + 1
        s0 = -torch.ones(self.words_per_seq, dtype=torch.long, device = torch.device(device_str))
        state_shape = s0.shape
        action_shape = (1,)
        dummy_action = -torch.ones(1, dtype=torch.long)
        exit_action = (n_actions - 1) * torch.ones(1, dtype=torch.long)
        sf = (n_actions - 1) * torch.ones(self.words_per_seq, dtype=torch.long, device = torch.device(device_str))
        super(BitSequence, self).__init__(
            n_actions,
            s0,
            state_shape,
            action_shape,
            dummy_action,
            exit_action,
            sf,
        )
    @staticmethod
    def is_balanced_parentheses(tensor: torch.Tensor) -> torch.Tensor:
        batch_size, length = tensor.shape
        balance = torch.zeros(batch_size, device=tensor.device)  # Balance counter
        min_balance = torch.zeros(batch_size, device=tensor.device)  # Track min balance

        for i in range(length):
            balance += torch.where(tensor[:, i] == 0, 1, -1)  # +1 for '(', -1 for ')'
            min_balance = torch.minimum(min_balance, balance)  # Track if balance ever goes negative
        
        # If the last element is -1, mark the row as unbalanced
        has_negative_one = tensor[:, -1] == -1
        return ((balance == 0) & (min_balance >= 0)) & (~has_negative_one)

    def update_masks(self, states: BitSequenceStates) -> None:
        """Updates the masks in States.

        Called automatically after each step for discrete environments.
        """
    
        is_done = (states.length== self.words_per_seq)
        states.forward_masks = torch.ones_like(states.forward_masks, dtype=torch.bool, device= states.__class__.device)
        states.forward_masks[is_done, :-1] = False

        is_sink = states.is_sink_state

        last_actions = states.tensor[~is_sink, states[~is_sink].length - 1]
        states.backward_masks = torch.zeros_like(states.backward_masks, dtype=torch.bool, device= states.__class__.device)
        states.backward_masks[~is_sink, last_actions] = True 

    def log_reward(self, final_states: BitSequenceStates):
        binary_tensor = self.integers_to_binary(final_states.tensor, self.word_size)
        is_under_8 = (final_states.length * self.word_size <= 8)
        is_balanced = self.is_balanced_parentheses(binary_tensor)
        log_rewards = -10. * torch.ones_like(is_balanced)
        log_rewards[is_under_8] = -1.5
        log_rewards[is_balanced] = 0.
        return log_rewards

    def reward(self, final_states: BitSequenceStates):
        return torch.exp(self.log_reward(final_states))

