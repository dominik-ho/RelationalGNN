from functools import cached_property
from typing import Any
import itertools

import torch
import torch_geometric
from torch import Tensor
from torch.nn import Module, Sigmoid, Identity
from torch_geometric.typing import EdgeType

import utils
from .atom_to_object_mp import AtomToObjectMP
from .object_to_atom_mp import ObjectToAtomMP
from .smooth_round import SmoothRound


class RelationalGNN(Module):
    def __init__(
            self,
            predicate_arity_dict: dict[str, int],
            object_type_name: str,
            num_layers: int | None,
            dynamic_num_layers: bool,
            embedding_size: int,
            activation: str,
            aggregation: str,
            manipulate_embeddings: None | str | tuple[str, Any]
    ):
        super().__init__()
        print("Initializing RelationalGNN...")
        self.predicate_arity_dict = {
            f"___{k}": v for k, v in predicate_arity_dict.items()  # hack to allow for predicate names such as "clear", which would else be forbidden
        }
        self.normal_to_safe_pred_map = {
            k: f"___{k}" for k in predicate_arity_dict.keys()
        }
        self.safe_to_normal_pred_map = {
            v: k for k, v in self.normal_to_safe_pred_map.items()
        }
        self.object_type_name = object_type_name
        # these dicts use the "normal" predicate names in their edge types
        self.objects_to_atoms_edge_types: list[EdgeType] = [
            (self.object_type_name, str(pos), pred)
            for pred, ar in predicate_arity_dict.items()
            for pos in range(ar)
        ]
        self.atoms_to_objects_edge_types: list[EdgeType] = [
            (pred, str(pos), self.object_type_name)
            for pred, ar in predicate_arity_dict.items()
            for pos in range(ar)
        ]


        if num_layers is None:
            if dynamic_num_layers:
                print("Setting num_layers to unbounded")
                self.layer_iterator = itertools.count(0)
            else:
                raise ValueError("Number of layers not specified, not using dynamic_num_layers")
        else:
            self.layer_iterator = range(num_layers)
        self.dynamic_num_layers = dynamic_num_layers
        self.embedding_size = embedding_size
        self.activation = activation
        self.aggregation = aggregation

        self.embedding_manipulator = RelationalGNN.match_embedding_manipulator(manipulate_embeddings)

        assert self.object_type_name not in self.predicate_arity_dict, f"<<{self.object_type_name}>> used for both predicate name and object type"
        for pred, ar in self.predicate_arity_dict.items():
            if ar == 0:
                print(f"removing 0 arity predicate {pred}")
                self.predicate_arity_dict.pop(pred)


        # init MLPs
        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self.embedding_size, 2 * self.embedding_size),
            torch.nn.ReLU(), # self.activation,
            torch.nn.Linear(2 * self.embedding_size, self.embedding_size),
        )

        # edge types are
        # object_type_name --pos-> predicate
        # and predicate --pos-> object_type_name
        self.predicate_mlps = torch.nn.ModuleDict({
            pred: torch.nn.Sequential(
                torch.nn.Linear(ar * self.embedding_size, ar *  self.embedding_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ar * self.embedding_size, ar * self.embedding_size),
            )
            for pred, ar in self.predicate_arity_dict.items()
        })

        # make all messages o_i->q(o_0, ..., o_(n-1)) in the shape k*(n-1), where for all j!=i the message is 0, and m_i is the message
        # then, can aggregate via sum and .view(-1)
        self.object_to_atom_mp = ObjectToAtomMP(self.embedding_size)
        self.atom_to_object_mp = AtomToObjectMP(self.embedding_size)
        # self.objects_to_atoms_hetero_conv = HeteroConv(convs={
        #     # for example:
        #     (self.object_type_name, "0", "on"): object_to_atom_mp,
        # }, aggr=None)

        # self.object_to_atom_mp_static_args_dict = {
        #     (self.object_type_name, str(pos), pred): pos
        #     for pred, ar in predicate_arity_dict.items()
        #     for pos in range(ar)
        # }

    @staticmethod
    def match_embedding_manipulator(manipulate_embeddings: None | str | tuple[str, Any]) -> Module:
        match manipulate_embeddings:
            case None:
                return Identity()
            case "sigmoid":
                return Sigmoid()
            case ("smooth_round", k):
                return SmoothRound(k)
            case _:
                raise ValueError(f"Unknown manipulator {manipulate_embeddings}")

    # @cached_property
    # def objects_to_atoms_edge_types(self) -> list[EdgeType]:
    #     return [
    #         (self.object_type_name, str(pos), pred)
    #         for pred, ar in self.predicate_arity_dict.items()
    #         for pos in range(ar)
    #     ]

    # @cached_property
    # def atoms_to_objects_edge_types(self) -> list[EdgeType]:
    #     return [
    #         (pred, str(pos), self.object_type_name)
    #         for pred, ar in self.predicate_arity_dict.items()
    #         for pos in range(ar)
    #     ]


    def init_embeddings(self, x_dict: dict[str, Tensor]):
        x_dict[self.object_type_name] = torch.zeros(
            (x_dict[self.object_type_name].shape[0], self.embedding_size),
            dtype=x_dict[self.object_type_name].dtype,
        )
        for pred, ar in self.predicate_arity_dict.items():
            normal_pred = self.safe_to_normal_pred_map[pred]
            x_dict[normal_pred] = torch.empty(
                (x_dict[normal_pred].shape[0], ar * self.embedding_size), dtype=x_dict[normal_pred].dtype
            )

    def layer(self, x_dict: dict[str, Tensor], edge_index_dict: dict[EdgeType, Tensor], num_objects: int | None, *args: Any, **kwargs: Any):
        # x_dict = self.objects_to_atoms_hetero_conv.forward(
        #     x_dict,
        #     edge_index_dict,
        #     self.object_to_atom_mp_static_args_dict
        # )
        if num_objects is None:
            num_objects = x_dict[self.object_type_name].shape[0]
        for pred, ar in self.predicate_arity_dict.items():
            normal_pred = self.safe_to_normal_pred_map[pred]
            pos_indexer = 0
            for pos in range(ar):
                edge_type: EdgeType = (self.object_type_name, str(pos), normal_pred)
                atom_embeddings_for_pos = self.object_to_atom_mp.forward(
                    x_dict[self.object_type_name],
                    edge_index_dict[edge_type],
                    num_atoms=x_dict[normal_pred].shape[0],
                )
                p = pos_indexer + self.embedding_size
                x_dict[normal_pred][:,pos_indexer:p] = atom_embeddings_for_pos # todo: auch hier testen, ob an richtiger stelle gesetzt wird
                pos_indexer = p
            x_dict[normal_pred] = self.predicate_mlps[pred](x_dict[normal_pred])

        # other direction:
        incoming_messages: list[Tensor] = []
        num_received_messages_per_object = torch.zeros(num_objects, dtype=torch.long)
        for edge_type in self.atoms_to_objects_edge_types:
            pred_type, pos, _ = edge_type
            messages = self.atom_to_object_mp.forward(
                x_dict[pred_type],
                edge_index_dict[edge_type],
                int(pos),
                num_objects=num_objects,
            )
            # are the messages for the object nodes in the same order as in edge_index_dict?
            num_received_messages_per_object += edge_index_dict[edge_type][1].bincount(minlength=num_objects)
            # edge_index_dict[edge_type][1] == object indices of the messages?
            incoming_messages.append(messages)
        tensored = torch.tensor(incoming_messages)
        # todo:
        aggregated_messages = self.aggregate(incoming_messages)
        stacked_for_update = torch.hstack((x_dict[self.object_type_name], aggregated_messages))
        if self.residual_updates:
            x_dict[self.object_type_name] = x_dict[self.object_type_name] + self.update_mlp(stacked_for_update)
        else:
            x_dict[self.object_type_name] = self.update_mlp(stacked_for_update)

        x_dict[self.object_type_name] = self.embedding_manipulator(x_dict[self.object_type_name])

        # return x_dict[self.object_type_name]

    def forward(self, x_dict: dict[str, Tensor], edge_index_dict: dict[EdgeType, Tensor], batch_assignment: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        self.init_embeddings(x_dict)

        if self.dynamic_num_layers:  # todo: test
            object_counts = torch.bincount(batch_assignment)

            sub_x_dict = x_dict
            sub_edge_index_dict = edge_index_dict
            for l in self.layer_iterator:
                relevant_states_mask = object_counts > (l + 1)
                if relevant_states_mask.max() < 1:
                    print(f"Stopping in iteration {l}")
                    break

                relevant_objects_mask = torch.repeat_interleave(relevant_states_mask, repeats=object_counts, dim=0)
                num_objects: int = relevant_objects_mask.sum(dtype=torch.int64).item()
                num_removed_objects = torch.logical_not(relevant_objects_mask).cumsum(dim=0, dtype=torch.long)
                relevant_objects_indices = torch_geometric.utils.mask_to_index(relevant_objects_mask)
                sub_x_dict[self.object_type_name] = sub_x_dict[self.object_type_name][relevant_objects_mask]

                for edge_type in self.atoms_to_objects_edge_types:
                    # for new_index, old_index in enumerate(relevant_objects_indices):
                    relevant_edges_mask = torch.isin(sub_edge_index_dict[edge_type][1], relevant_objects_indices)
                    sub_edge_index_dict[edge_type] = sub_edge_index_dict[edge_type][:, relevant_edges_mask]
                    sub_edge_index_dict[edge_type][1] -= num_removed_objects[sub_edge_index_dict[edge_type][1]]

                    relevant_atoms_indices = torch.unique(sub_edge_index_dict[edge_type][0])
                    relevant_atoms_mask = torch_geometric.utils.index_to_mask(relevant_atoms_indices)
                    num_removed_atoms = torch.logical_not(relevant_atoms_mask).cumsum(dim=0, dtype=torch.long)
                    sub_edge_index_dict[edge_type][0] -= num_removed_atoms[sub_edge_index_dict[edge_type][0]]

                    sub_edge_index_dict[utils.invert_edge_type(edge_type)] = sub_edge_index_dict[edge_type].flip(0)
                    # relevant_atoms_mask = torch.zeros(sub_x_dict[edge_type[0]].size(0), dtype=torch.bool).index_fill(dim=0, index=sub_edge_index_dict[edge_type][0], value=1)
                    # sub_x_dict[edge_type[0]] = sub_x_dict[edge_type[0]][relevant_atoms_indices]

                    # since we don't care about the actual embeddings of atoms at this point, just
                    # shrink the sub_x_dict enough without keeping exactly the relevant embeddings
                    num_remaining_atoms = relevant_atoms_indices.size(0)
                    sub_x_dict[edge_type[0]] = sub_x_dict[edge_type[0]][:num_remaining_atoms]
                self.layer(sub_x_dict, sub_edge_index_dict, num_objects=num_objects, *args, **kwargs)  # x_dict[self.object_type_name] =
        else:
            num_objects = x_dict[self.object_type_name].shape[0]
            for l in self.layer_iterator:
                self.layer(x_dict, edge_index_dict, num_objects=num_objects, *args, **kwargs)  # x_dict[self.object_type_name] =

        return x_dict[self.object_type_name]

class MLPReadoutRGNN(RelationalGNN):
    def __init__(
            self,
            predicate_arity_dict: dict[str, int],
            object_type_name: str,
            num_layers: int | None,
            dynamic_num_layers: bool,
            embedding_size: int,
            activation: str,
            aggregation: str,
            manipulate_embeddings: None | str | tuple[str, Any]
    ):
        super().__init__(predicate_arity_dict, object_type_name, num_layers, dynamic_num_layers, embedding_size,
                         activation, aggregation, manipulate_embeddings)
        self.state_agg: torch_geometric.nn.Aggregation = torch_geometric.nn.SumAggregation()
        self.readout_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, 2 * self.embedding_size),
            torch.nn.ReLU(), # self.readout_activation,
            torch.nn.Linear(2 * self.embedding_size, 1),
        )

    def forward(self, x_dict: dict[str, Tensor], edge_index_dict: dict[EdgeType, Tensor], batch_assignment: Tensor | None, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        if batch_assignment is None:
            batch_assignment = torch.zeros(x_dict[self.object_type_name].size(0), dtype=torch.long)
        object_embeddings = super().forward(x_dict, edge_index_dict, batch_assignment, *args, **kwargs)
        # statewise_object_embeddings = torch_geometric.utils.unbatch(object_embeddings, batch_assignment)
        state_embeddings = self.state_agg(object_embeddings, batch_assignment)
        return self.readout_mlp(state_embeddings)

class FeatureAlignedReadoutRGNN(RelationalGNN):
    def __init__(
            self,
            predicate_arity_dict: dict[str, int],
            object_type_name: str, num_layers: int | None,
            dynamic_num_layers: bool,
            embedding_size: int,
            activation: str,
            aggregation: str,
            manipulate_embeddings: None | str | tuple[str, Any]
    ):
        super().__init__(predicate_arity_dict, object_type_name, num_layers, dynamic_num_layers, embedding_size,
                         activation, aggregation, manipulate_embeddings)
        self.state_agg_sum: torch_geometric.nn.Aggregation = torch_geometric.nn.SumAggregation()
        self.state_agg_max: torch_geometric.nn.Aggregation = torch_geometric.nn.MaxAggregation()
        self.readout_lin = torch.nn.Linear(2 * self.embedding_size, 1)

    def forward(self, x_dict: dict[str, Tensor], edge_index_dict: dict[EdgeType, Tensor], batch_assignment: Tensor | None, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        if batch_assignment is None:
            batch_assignment = torch.zeros(x_dict[self.object_type_name].size(0), dtype=torch.long)
        object_embeddings = super().forward(x_dict, edge_index_dict, batch_assignment, *args, **kwargs)
        # statewise_object_embeddings = torch_geometric.utils.unbatch(object_embeddings, batch_assignment)
        state_embeddings_sum = self.state_agg_sum(object_embeddings, batch_assignment)
        state_embeddings_max = self.state_agg_max(object_embeddings, batch_assignment)
        state_embeddings = torch.hstack(
            (state_embeddings_sum, state_embeddings_max)
        )
        return self.readout_lin(state_embeddings)
