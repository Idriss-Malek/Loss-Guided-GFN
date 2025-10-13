import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from torch import nn
from torch_geometric.data import Batch as GeometricBatch

from gfn.utils.common import set_seed


from src.algos import train_tb, train_lggfn, train_sagfn, train_AT
from src.gflownet import TBGFlowNetV2
from gfn.gym.helpers.bayesian_structure.factories import get_scorer
from gfn.gym.bayesian_structure import BayesianStructure
from gfn.gym.helpers.bayesian_structure.evaluation import (
    expected_edges,
    expected_shd,
    posterior_estimate,
    threshold_metrics,
)
from argparse import Namespace, ArgumentParser

import torch
from tensordict import TensorDict
from torch import nn
from torch_geometric.data import Batch as GeometricBatch

from gfn.actions import GraphActionType

from gfn.modules import DiscreteGraphPolicyEstimator
from gfn.utils.common import set_seed
from gfn.utils.modules import GraphActionUniform, GraphEdgeActionGNN, GraphEdgeActionMLP

DEFAULT_SEED = 4444


class DAGEdgeActionMLP(GraphEdgeActionMLP):

    def __init__(
        self,
        n_nodes: int,
        num_edge_classes: int,
        n_hidden_layers: int = 2,
        n_hidden_layers_exit: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        super().__init__(
            n_nodes=n_nodes,
            directed=True,
            num_edge_classes=num_edge_classes,
            n_hidden_layers=n_hidden_layers,
            n_hidden_layers_exit=n_hidden_layers_exit,
            embedding_dim=embedding_dim,
            is_backward=is_backward,
        )

    @property
    def edges_dim(self) -> int:
        return self.n_nodes**2


class DAGEdgeActionGNN(GraphEdgeActionGNN):
    """Simple module which outputs a fixed logits for the actions, depending on the number of nodes.
    Args:
        n_nodes: The number of nodes in the graph.
    """

    def __init__(
        self,
        n_nodes: int,
        num_edge_classes: int,
        num_conv_layers: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        super().__init__(
            n_nodes=n_nodes,
            directed=True,
            num_edge_classes=num_edge_classes,
            num_conv_layers=num_conv_layers,
            embedding_dim=embedding_dim,
            is_backward=is_backward,
        )

    @property
    def edges_dim(self) -> int:
        return self.n_nodes**2

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        node_features, batch_ptr = (states_tensor.x, states_tensor.ptr)

        # Multiple action type convolutions with residual connections.
        x = self.embedding(node_features.squeeze().int())
        for i in range(0, len(self.conv_blks), 2):
            x_new = self.conv_blks[i](x, states_tensor.edge_index)  # GIN/GCN conv.
            assert isinstance(self.conv_blks[i + 1], nn.ModuleList)
            x_in, x_out = torch.chunk(x_new, 2, dim=-1)

            # Process each component separately through its own MLP.
            mlp_in, mlp_out = self.conv_blks[i + 1]
            x_in = mlp_in(x_in)
            x_out = mlp_out(x_out)
            x_new = torch.cat([x_in, x_out], dim=-1)

            x = x_new + x if i > 0 else x_new  # Residual connection.
            x = self.norm(x)  # Layernorm.

        # This MLP computes the exit action.
        node_feature_means = self._group_mean(x, batch_ptr)
        if not self.is_backward:
            exit_action = self.exit_mlp(node_feature_means).squeeze(-1)

        x = x.reshape(*states_tensor.batch_shape, self.n_nodes, self.hidden_dim)

        feature_dim = self.hidden_dim // 2
        source_features = x[..., :feature_dim]
        target_features = x[..., feature_dim:]

        # Dot product between source and target features (asymmetric).
        edgewise_dot_prod = torch.einsum(
            "bnf,bmf->bnm", source_features, target_features
        )
        edgewise_dot_prod = edgewise_dot_prod / torch.sqrt(torch.tensor(feature_dim))

        # Grab the needed elems from the adjacency matrix and reshape.
        edge_actions = edgewise_dot_prod.flatten(1, 2)
        assert edge_actions.shape == (*states_tensor.batch_shape, self.edges_dim)

        action_type = torch.zeros(*states_tensor.batch_shape, 3, device=x.device)
        if self.is_backward:
            action_type[..., GraphActionType.ADD_EDGE] = 1
        else:
            action_type[..., GraphActionType.ADD_EDGE] = 1 - exit_action
            action_type[..., GraphActionType.EXIT] = exit_action

        return TensorDict(
            {
                "action_type": action_type,
                "edge_class": torch.zeros(
                    *states_tensor.batch_shape, self.num_edge_classes, device=x.device
                ),  # TODO: make it learnable.
                "node_class": torch.zeros(
                    *states_tensor.batch_shape, 1, device=x.device
                ),
                "edge_index": edge_actions,
            },
            batch_size=states_tensor.batch_shape,
        )


def main(args: Namespace):
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    rng = np.random.default_rng(seed)  # This should be cpu

    # Create the scorer
    scorer, _, gt_graph = get_scorer(
        args.graph_name,
        args.prior_name,
        args.num_nodes,
        args.num_edges,
        args.num_samples,
        args.node_names,
        rng=rng,
    )

    # Create the environment
    env = BayesianStructure(
        n_nodes=args.num_nodes,
        state_evaluator=scorer.state_evaluator,
        device=device,
    )

    if args.use_gnn:
        pf_module = DAGEdgeActionGNN(
            env.n_nodes,
            env.num_edge_classes,
            args.num_layers,
            args.embedding_dim,
        )
    else:
        pf_module = DAGEdgeActionMLP(
            env.n_nodes,
            env.num_edge_classes,
            args.num_layers,
            1,
            args.embedding_dim,
        )

    pb_module = GraphActionUniform(
        env.n_actions - 1,  # equivalent to env.n_nodes**2
        env.num_edge_classes,
        env.num_node_classes,
    )
    pf = DiscreteGraphPolicyEstimator(
        module=pf_module,
    )
    pb = DiscreteGraphPolicyEstimator(
        module=pb_module,
        is_backward=True,
    )
    mainGFN = TBGFlowNetV2(pf, pb)
    mainGFN = mainGFN.to(device)

    # Log Z gets dedicated learning rate (typically higher).
    params = [
        {
            "params": [
                v for k, v in dict(mainGFN.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]
    if "logZ" in dict(mainGFN.named_parameters()):
        params.append(
            {
                "params": [dict(mainGFN.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )

    # -------------------- Auxiliary module ------------------------------------
    if args.use_gnn:
        pf_module_aux = DAGEdgeActionGNN(
            env.n_nodes,
            env.num_edge_classes,
            args.num_layers,
            args.embedding_dim,
        )
    else:
        pf_module_aux = DAGEdgeActionMLP(
            env.n_nodes,
            env.num_edge_classes,
            args.num_layers,
            1,
            args.embedding_dim,
        )

    pb_module_aux = GraphActionUniform(
        env.n_actions - 1,  # equivalent to env.n_nodes**2
        env.num_edge_classes,
        env.num_node_classes,
    )
    pf_aux = DiscreteGraphPolicyEstimator(
        module=pf_module_aux,
    )
    pb_aux = DiscreteGraphPolicyEstimator(
        module=pb_module_aux,
        is_backward=True,
    )
    auxGFN = TBGFlowNetV2(pf_aux, pb_aux)
    auxGFN = auxGFN.to(device)

    # Log Z gets dedicated learning rate (typically higher).
    params = [
        {
            "params": [
                v for k, v in dict(mainGFN.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]
    if "logZ" in dict(mainGFN.named_parameters()):
        params.append(
            {
                "params": [dict(mainGFN.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )

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
    elif args.algo == "AT":
        train_AT(
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


    # Compute the metrics
    with torch.no_grad():
        posterior_samples = posterior_estimate(
            mainGFN,
            env,
            num_samples=args.num_samples_posterior,
            batch_size=args.batch_size,
        )
    shd = expected_shd(posterior_samples, gt_graph)
    print(f"Expected SHD: {expected_shd(posterior_samples, gt_graph)}")
    print(f"Expected edges: {expected_edges(posterior_samples)}")
    thres_metrics = threshold_metrics(posterior_samples, gt_graph)
    for k, v in thres_metrics.items():
        print(f"{k}: {v}")
    edge_prob = args.num_edges / (args.num_nodes * (args.num_nodes - 1) / 2)
    results_file = "lggfn_results.txt"
    with open(results_file, "a+") as f:
        f.write(f"{args.num_nodes} Nodes with edge probability {edge_prob}  and seed {args.seed}| "
                f"ROC AUC: {thres_metrics['roc_auc']:.4f} | "
                f"SHD: {shd:.4f}\n")


if __name__ == "__main__":
    parser = ArgumentParser(
        "Train a GFlowNet to generate a DAG for Bayesian structure learning."
    )
    # Environment parameters
    parser.add_argument("--num_nodes", type=int, default=5)
    parser.add_argument(
        "--num_edges",
        type=int,
        default=5,
        help="Number of edges in the sampled erdos renyi graph",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples."
    )
    parser.add_argument("--graph_name", type=str, default="erdos_renyi_lingauss")
    parser.add_argument("--prior_name", type=str, default="uniform")
    parser.add_argument(
        "--node_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of node names.",
    )
    parser.add_argument(
        "--num_samples_posterior",
        type=int,
        default=1000,
        help="Number of samples for posterior approximation.",
    )

    # GFlowNet and policy parameters
    parser.add_argument("--use_gnn", action="store_true")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--max_epsilon", type=float, default=0.9)
    parser.add_argument("--min_epsilon", type=float, default=0.1)


    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_Z", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--algo', type=str, default="tb", choices=["tb", "lggfn", "sagfn", "AT"])
    parser.add_argument('--device_str', type=str, default="cpu")

    # SAGFN parameters
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--beta_e', type=float, default=1.0)
    parser.add_argument('--beta_i', type=float, default=1.0)
    parser.add_argument('--beta_sn', type=float, default=1.0)
    parser.add_argument('--rnd_lr', type=float, default=1e-3)

    #LGGFN parameters
    parser.add_argument('--lamda', type=float, default=1.0)


    # Misc parameters
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
