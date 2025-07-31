import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from datetime import datetime
import yaml
from types import SimpleNamespace
import logging
import time
import torch
import wandb
import torch.nn as nn 
import numpy as np

from env import CodonDesignEnv
from utils import *

from gfn.utils.modules import MLP
from gfn.modules import DiscretePolicyEstimator
from gfn.states import States
from gfn.preprocessors import Preprocessor

from src.gflownet import TBGFlowNetV2
from src.algos import train_tb, train_lggfn, train_sagfn

from torchgfn.src.gfn.samplers import Sampler
from evaluate import evaluate


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class CodonSequencePreprocessor(Preprocessor):
    """Preprocessor for codon sequence states"""
    def __init__(self, seq_length: int, embedding_dim: int, device: torch.device):
        super().__init__(output_dim=seq_length * embedding_dim)
        self.device = device
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(N_CODONS + 1, embedding_dim, padding_idx=N_CODONS).to(device)  
               
    def preprocess(self, states: States) -> torch.Tensor:
        states_tensor = states.tensor.long().clone()
        states_tensor[states_tensor == -1] = N_CODONS
        states_tensor = states_tensor.to(self.device)
        embedded = self.embedding(states_tensor)
        out = embedded.view(states_tensor.shape[0], -1)
        return out


def main(args, config):

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    if args.wandb_project:

        logging.info("Initializing Weights & Biases...")
        wandb.init(
            project = config.wandb_project,
            config = {**vars(args), **vars(config)},
            name = args.run_name 
        )

        if wandb.run is not None:
            wandb.define_metric("iteration")
            for metric in ["loss", "reward", "gc", "mfe", "cai"]:
                wandb.define_metric(metric, step_metric="iteration")

    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor(env.seq_length, embedding_dim=args.embedding_dim, device=device)

    # Build main PF/PB networks
    mainPF = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions, hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden)
    mainPB = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions-1, hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden, trunk=mainPF.trunk if args.tied else None)
    
    mainPF_est = DiscretePolicyEstimator(mainPF, env.n_actions, is_backward=False, preprocessor=preprocessor)
    mainPB_est = DiscretePolicyEstimator(mainPB, env.n_actions, is_backward=True, preprocessor=preprocessor)
    
    mainGFN = TBGFlowNetV2(pf=mainPF_est, pb=mainPB_est, logZ=0.).to(device)

    # Build aux PF/PB networks
    auxPF = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions, hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden)
    auxPB = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions-1, hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden, trunk=auxPF.trunk if args.tied else None)
    
    auxPF_est = DiscretePolicyEstimator(auxPF, env.n_actions, is_backward=False, preprocessor=preprocessor)
    auxPB_est = DiscretePolicyEstimator(auxPB, env.n_actions, is_backward=True, preprocessor=preprocessor)
    
    auxGFN = TBGFlowNetV2(pf=auxPF_est, pb=auxPB_est, logZ=0.).to(device)

    sampler = Sampler(estimator=mainPF_est)

    optimizer = torch.optim.Adam(mainGFN.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": mainGFN.logz_parameters(), "lr": args.lr_logz})
    logging.info("Starting training loop...")

    t0 = time.time()

    if args.algo == "tb":
        history = train_tb(env, mainGFN, batch_size=args.batch_size, iterations=args.n_iterations, lr=args.lr,
                           lr_Z=args.lr_logz, device_str=str(device))
    elif args.algo == "lggfn":
        history = train_lggfn(env, mainGFN, auxGFN, batch_size=args.batch_size, iterations=args.n_iterations,
                              lamda=args.lamda, lr=args.lr, lr_Z=args.lr_logz, device_str=str(device))
    elif args.algo == "sagfn":
        history = train_sagfn(env, mainGFN, auxGFN, batch_size=args.batch_size, iterations=args.n_iterations,
                               reward_scale=args.reward_scale, beta_e=args.beta_e, beta_i=args.beta_i,
                               beta_sn=args.beta_sn, lr=args.lr, lr_Z=args.lr_logz,
                               lr_rnd=args.rnd_lr, device_str=str(device))

    train_time = time.time() - t0
    logging.info(f"Training completed in {train_time:.2f} s")
    logging.info("Starting inference sampling...")

    t1 = time.time()
    with torch.no_grad(): 
        samples, gc, mfe, cai = evaluate(env, mainGFN, weights=torch.tensor(config.reward_weights), n_samples=args.n_samples)   #mainGFN
    
    inference_time = time.time() - t1
    avg_time_per_seq = inference_time / args.n_samples
    logging.info(f"Inference completed in {inference_time:.2f} s")

    ################ Final means over the evaluation samples ###############

    eval_mean_gc  = float(torch.tensor(gc).mean())
    eval_mean_mfe = float(torch.tensor(mfe).mean())
    eval_mean_cai = float(torch.tensor(cai).mean())

    plot_metric_histograms(gc, mfe, cai, out_path=f"metric_distributions_for_algo_{args.algo}.png")
    sorted_samps = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

    filename = f"generated_sequences_for_algo_{args.algo}.txt"

    with open(filename, "w") as f:

        for i, (seq, (rew, comps)) in enumerate(sorted_samps[:args.top_n]):
            f.write(
                f"Sequence {i+1}: {seq}, "
                f"Reward: {rew:.2f}, "
                f"GC Content: {comps[0]:.2f}, "
                f"MFE: {comps[1]:.2f}, "
                f"CAI: {comps[2]:.2f}\n"
            )

    top_n = args.top_n    
    sequences = [seq for seq, _ in sorted_samps[:top_n]]
    distances = analyze_diversity(sequences, out_path=f"edit_distance_distribution_for_algo_{args.algo}.png") 

    generated_sequences_tensor = [
        tokenize_sequence_to_tensor(seq)
        for seq in sequences
    ]

    # Get best sequences for each reward component
    best_gc = max(samples.items(), key=lambda x: x[1][1][0])   # GC content
    best_mfe = min(samples.items(), key=lambda x: x[1][1][1])  # MFE
    best_cai = max(samples.items(), key=lambda x: x[1][1][2])  # CAI

    additional_best_seqs = [best_gc[0], best_mfe[0], best_cai[0]]

    for s in additional_best_seqs:
        generated_sequences_tensor.append(tokenize_sequence_to_tensor(s))

    natural_tensor = tokenize_sequence_to_tensor(config.natural_mRNA_seq)
    sequence_labels = [f"Gen {i+1}" for i in range(top_n)] + ["Best GC", "Best MFE", "Best CAI"]


    ############### Perform a comparison between sequences and the natural mRNA sequences ###############

    analyze_sequence_properties(
        generated_sequences_tensor,
        natural_tensor,
        labels=sequence_labels,
        run_name=args.algo
    )

    if args.wandb_project:

        wandb.log({
            'Train_time': train_time,
            'Inference_time': inference_time,
            "Avg_time_per_sequence": avg_time_per_seq,
            'Unique_sequences': len(samples),
            "Eval_mean_gc":  eval_mean_gc,
            "Eval_mean_mfe": eval_mean_mfe,
            "Eval_mean_cai": eval_mean_cai,
            "Mean_edit_distance": np.mean(distances),
            "Std_edit_distance": np.std(distances),
            "Edit_distance_distribution": wandb.Image(f"edit_distance_distribution_for_algo_{args.algo}.png"),
            "Reward Metric distributions": wandb.Image(f"metric_distributions_for_algo_{args.algo}.png"),
        })

        wandb.finish()    


if __name__ == "__main__":

    setup_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", choices=["tb","lggfn","sagfn"], default="tb")

    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Training parameters - GFlowNet and policy parameters
    parser.add_argument("--lr", type=float, default=1e-3) 
    parser.add_argument("--lr_logz", type=float, default=1e-1)
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--top_n", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=16) 

    parser.add_argument("--epsilon", type=float, default=0.2)  # for exploration
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden", type=int, default=2)   
    parser.add_argument("--tied", action="store_true")

    # SAGFN parameters
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--beta_e", type=float, default=1.0)
    parser.add_argument("--beta_i", type=float, default=1.0)
    parser.add_argument("--beta_sn", type=float, default=1.0)
    parser.add_argument("--rnd_lr", type=float, default=1e-2)

    #LGGFN parameters
    parser.add_argument("--lamda", type=float, default=1.0)

    parser.add_argument("--wandb_project", type=str, default="mRNA_design")
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument('--config_path', type=str, default="config.yaml")

    args = parser.parse_args()
    config = load_config(args.config_path)

    for algo in ["tb", "lggfn", "sagfn"]:

        args.algo = algo
        args.run_name = f"{algo}_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"\n>>> Running: {algo}")

        main(args, config)


  



















