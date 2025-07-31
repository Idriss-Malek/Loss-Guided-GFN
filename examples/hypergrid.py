import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.gym import HyperGrid
from gfn.utils.modules import MLP
from gfn.preprocessors import KHotPreprocessor
from gfn.modules import DiscretePolicyEstimator

from src.gflownet import TBGFlowNetV2
from src.algos import train_tb, train_lggfn, train_sagfn
from src.utils.hypergrid import plot_distribution, l1_error

def main(args):

    env = HyperGrid(height=args.height, ndim=args.ndim, R0=args.R0, R1=args.R1, R2=args.R2, calculate_all_states=True)
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    mainPF = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    mainPB = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        trunk=mainPF.trunk,
    )

    mainPF_estimator = DiscretePolicyEstimator(
        mainPF, env.n_actions, is_backward=False, preprocessor=preprocessor
    )
    mainPB_estimator = DiscretePolicyEstimator(
        mainPB, env.n_actions, is_backward=True, preprocessor=preprocessor
    )

    mainGFN = TBGFlowNetV2(pf=mainPF_estimator, pb=mainPB_estimator, logZ=0.)

    auxPF = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    auxPB = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        trunk=auxPF.trunk,
    )



    auxPF_estimator = DiscretePolicyEstimator(
        auxPF, env.n_actions, is_backward=False, preprocessor=preprocessor
    )
    auxPB_estimator = DiscretePolicyEstimator(
        auxPB, env.n_actions, is_backward=True, preprocessor=preprocessor
    )

    auxGFN = TBGFlowNetV2(pf=auxPF_estimator, pb=auxPB_estimator, logZ=0.)

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

    print("Training complete.")
    print('L1 error:', l1_error(mainGFN.pf, env))
    plot_distribution(mainGFN.pf, env)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ################ Hypergrid Environment ###############
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--ndim', type=int, default=2)
    parser.add_argument('--R0', type=float,default=1e-4)
    parser.add_argument('--R1', type=float,default=1.)
    parser.add_argument('--R2', type=float,default=3.)

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

    args = parser.parse_args()
    main(args)

