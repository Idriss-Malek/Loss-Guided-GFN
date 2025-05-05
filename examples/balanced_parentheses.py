import sys
import os
import argparse
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packaging.version import parse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2 import GPT2Model, GPT2Config

from tqdm import tqdm
from gfn.utils.common import set_seed


from gfn.modules import DiscretePolicyEstimator
from src.gflownet import TBGFlowNetV2
from gfn.samplers import Sampler

from gfn.preprocessors import Preprocessor
from src.envs import BalancedParentheses
from gfn.gym.bitSequence import BitSequenceStates
from src.algos import train_tb, train_lggfn, train_sagfn


class BitSequencesTransformerPreprocessor(Preprocessor):
    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    def preprocess(self, states: BitSequenceStates) -> Dict[str, torch.Tensor]:
        """Transform the states to the input of the Hugging Face GPT2 Transformer.

        Args:
            states: The states to preprocess.

        Returns the preprocessed states as a tensor of shape (*batch_shape, output_dim).
        """
        input_ids = states.tensor

        mask = (input_ids != -1).to(input_ids.dtype)
        input_ids[input_ids == -1] = 0
        return {'input_ids':input_ids,'attention_mask':mask}


    def __call__(self, states: BitSequenceStates) -> Dict[str, torch.Tensor]:
        """Transform the states to the input of the neural network, calling the preprocess method."""
        out = self.preprocess(states)
        assert out['input_ids'].shape[-1] == self.output_dim
        assert out['attention_mask'].shape[-1] == self.output_dim
        return out

class Transformer(nn.Module):
    def __init__(self, feature_extractor, head):
        super().__init__()
        self.fe = feature_extractor
        self.head : nn.Linear = head
    def forward(self,input:dict):
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        features = self.fe(input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.head(features)
        first_masked_indices = (attention_mask == 0).int().argmax(dim=1)
        first_masked_logits = logits[torch.arange(logits.shape[0]), first_masked_indices]
        return first_masked_logits

def main(args):
    set_seed(args.seed, performance_mode=True)
    device = torch.device(args.device_str)
    env = BalancedParentheses(seq_size=args.seq_size, word_size=args.word_size, device_str=args.device_str)
    config = GPT2Config(
        vocab_size=(2 ** env.word_size) + 1,  # Number of actions or discrete tokens
        n_positions=env.words_per_seq,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )
    preprocessor = BitSequencesTransformerPreprocessor(config.n_positions)

    main_feature_extractor = GPT2Model(config)
    aux_feature_extractor = GPT2Model(config)

    main_pf_head = nn.Linear(config.n_embd, config.vocab_size)
    main_pb_head = nn.Linear(config.n_embd, config.vocab_size - 1)
    aux_pf_head = nn.Linear(config.n_embd, config.vocab_size)
    aux_pb_head = nn.Linear(config.n_embd, config.vocab_size - 1)

    main_pf = Transformer(main_feature_extractor, main_pf_head)
    main_pb = Transformer(main_feature_extractor, main_pb_head)
    aux_pf = Transformer(aux_feature_extractor, aux_pf_head)
    aux_pb = Transformer(aux_feature_extractor, aux_pb_head)

    main_pf_estimator = DiscretePolicyEstimator(main_pf, n_actions=config.vocab_size, preprocessor=preprocessor)
    main_pb_estimator = DiscretePolicyEstimator(main_pb, n_actions=config.vocab_size, preprocessor=preprocessor, is_backward=True)
    aux_pf_estimator = DiscretePolicyEstimator(aux_pf, n_actions=config.vocab_size, preprocessor=preprocessor)
    aux_pb_estimator = DiscretePolicyEstimator(aux_pb, n_actions=config.vocab_size, preprocessor=preprocessor, is_backward=True)

    auxGFN = TBGFlowNetV2(pf=aux_pf_estimator, pb=aux_pb_estimator, logZ=0.).to(device)
    mainGFN = TBGFlowNetV2(pf=main_pf_estimator, pb=main_pb_estimator, logZ=0.).to(device)

    main_non_logz_params = [v for k, v in dict(mainGFN.named_parameters()).items() if k != "logZ"]
    mainOptimizer = torch.optim.Adam(main_non_logz_params, lr=1e-3)
    logz_params = [dict(mainGFN.named_parameters())["logZ"]]
    mainOptimizer.add_param_group({"params": logz_params, "lr": 1e-1})

    aux_non_logz_params = [v for k, v in dict(auxGFN.named_parameters()).items() if k != "logZ"]
    auxOptimizer = torch.optim.Adam(aux_non_logz_params, lr=1e-3)
    auxlogz_params = [dict(auxGFN.named_parameters())["logZ"]]
    auxOptimizer.add_param_group({"params": auxlogz_params, "lr": 1e-1})


    if args.algo == "tb":
        train_tb(
            env=env,
            mainGFN=mainGFN,
            batch_size=args.batch_size,
            iterations=args.iterations,
            lr=args.lr,
            lr_Z=args.lr_Z,
            device_str=args.device_str
        )
    elif args.algo == "lggfn":
        train_lggfn(
            env=env,
            mainGFN=mainGFN,
            auxGFN=auxGFN,
            batch_size=args.batch_size,
            iterations=args.iterations,
            lamda=args.lamda,
            lr=args.lr,
            lr_Z=args.lr_Z,
            device_str=args.device_str
        )
    elif args.algo == "sagfn":
        train_sagfn(
            env=env,
            mainGFN=mainGFN,
            auxGFN=auxGFN,
            batch_size=args.batch_size,
            iterations=args.iterations,
            reward_scale=args.reward_scale,
            beta_e=args.beta_e,
            beta_i=args.beta_i,
            beta_sn=args.beta_sn,
            lr=args.lr,
            lr_Z=args.lr_Z,
            lr_rnd=args.rnd_lr,
            device_str=args.device_str
        )

    
    mainGFN.eval()
    is_balanced = []
    state_tensors = []
    with torch.no_grad():
        for _ in range(1000):
            last_states = mainGFN.sample_trajectories(env=env, n=args.batch_size, save_logprobs=False).terminating_states.tensor.cpu()
            is_balanced.append(env.is_balanced_parentheses(last_states))
            state_tensors.append(last_states)

        is_balanced = torch.cat(is_balanced, dim=0)
        state_tensors = torch.cat(state_tensors, dim=0)
        rate_of_complete_balanced_parentheses = torch.mean(is_balanced.float())
        unique_balanced_parentheses = torch.unique(state_tensors[is_balanced], dim=0).shape[0]
    return rate_of_complete_balanced_parentheses, unique_balanced_parentheses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ################ Hypergrid Environment ################
    parser.add_argument('--seq_size', type=int, default=32)
    parser.add_argument('--word_size', type=int, default=1)

    ################ Training Hyperparameters #############
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_Z', type=float, default=1e-1)
    parser.add_argument('--device_str', type=str, default="cpu")
    parser.add_argument('--algo', type=str, default="tb", choices=["tb", "lggfn", "sagfn"])

    ################## SAGFN Hyperparameters #############
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--beta_e', type=float, default=1.0)
    parser.add_argument('--beta_i', type=float, default=1.0)
    parser.add_argument('--beta_sn', type=float, default=1.0)
    parser.add_argument('--rnd_lr', type=float, default=1e-3)

    ################## LGGFN Hyperparameters #############
    parser.add_argument('--lamda', type=float, default=1.0)

    ################## Transformer Hyperparameters #############
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=4)

    ################## Miscellaneous #############
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    rate, count = main(args)
    print('Rate of complete balanced parentheses:', rate.item())
    print('Unique balanced parentheses:', count)
