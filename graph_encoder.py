from functools import cache

import torch
from networkx import MultiDiGraph
import pymimir
from pymimir import State, Object, Domain, GroundAtom, GroundLiteral
from torch_geometric.data import HeteroData

from utils import get_predicate_arity_dict


class GraphEncoder:
    ignored_predicates = ["object", "number"]

    def __init__(
            self,
            domain: Domain,
            object_type_name: str,
            predicate_arity_dict: dict[str, int] | None = None,
    ):
        self.domain = domain
        self.object_type_name = object_type_name
        self.predicate_arity_dict = predicate_arity_dict or get_predicate_arity_dict(self.domain.get_predicates())
        self.all_used_predicate_names = {
            p for pred in self.domain.get_predicates()
            if (p:=pred.get_name()) not in GraphEncoder.ignored_predicates
        }
        self.pos_goal_predicate_names_dict = {
            p: self.add_goal_postfix(pred.get_name())
            for pred in self.domain.get_predicates()
            if (p := pred.get_name()) not in GraphEncoder.ignored_predicates
        }
        self.neg_goal_predicate_names_dict = {
            name: self.add_neg_prefix(self.pos_goal_predicate_names_dict[name])
            for name, goal_pred_name in self.pos_goal_predicate_names_dict.items()
        }
        self.all_used_predicate_names |= set(self.pos_goal_predicate_names_dict.values()) | set(self.neg_goal_predicate_names_dict.values())

    def encode(self, state: State, objects: list[Object]) -> MultiDiGraph:
        result = MultiDiGraph()
        result.add_nodes_from(objects, node_type=self.object_type_name)  # [o.get_name() for o in objects]
        for atom in state.get_atoms():
            if atom.get_predicate().get_name() in GraphEncoder.ignored_predicates:
                continue
            result.add_node(atom, node_type=atom.get_predicate().get_name(), class_type=GroundAtom)
            for i, o in enumerate(atom.get_terms()):
                result.add_edges_from(((o, atom), (atom, o)), pos=i)
        for goal_literal in state.get_problem().get_goal_condition().get_literals():
            atom = goal_literal.get_atom()
            name = atom.get_predicate().get_name()
            if name in GraphEncoder.ignored_predicates:
                continue
            is_negated = not goal_literal.get_polarity()
            if is_negated:
                # t = self.add_neg_prefix(self.add_goal_postfix(atom.get_predicate().get_name()))
                t = self.neg_goal_predicate_names_dict[name]
            else:
                # t = self.add_goal_postfix(atom.get_predicate().get_name())
                t = self.pos_goal_predicate_names_dict[name]
            result.add_node(goal_literal, node_type=t, class_type=GroundLiteral)
            for i, o in enumerate(atom.get_terms()):
                result.add_edges_from(((o, atom), (atom, o)), pos=i)
        return result

    # @cache
    def add_goal_postfix(self, predicate_name: str):
        # result = self.goal_predicate_names_dict.get(predicate_name)
        # if result is not None:
        #     return result

        postfix = "_G"
        result = predicate_name + postfix
        while result in self.all_used_predicate_names:
            result += postfix
        self.all_used_predicate_names.add(result)
        return result

    # @cache
    def add_neg_prefix(self, predicate_name: str):
        # result = self.goal_predicate_names_dict.get(predicate_name)
        # if result is not None:
        #     return result

        prefix = "not:"
        result = prefix + predicate_name
        while result in self.all_used_predicate_names:
            result = result + prefix
        self.all_used_predicate_names.add(result)
        return result

    def to_pyg(self, graph: MultiDiGraph, **kwargs) -> HeteroData:
        result = HeteroData()

        objects_list: list[Object] = []
        pyg_atom_nodes = {
            p: [] for p in self.all_used_predicate_names
        }
        # {self.object_type_name: []}

        for node, data in graph.nodes.items():
            node_type = data["node_type"]
            if node_type == self.object_type_name:
                objects_list.append(node)
                continue
            # node is GroundAtom or goal literal (GroundLiteral)
            pyg_atom_nodes[node_type].append(node)

        result[self.object_type_name].x = torch.zeros(len(objects_list))
        result.obj_names = [o.get_name() for o in objects_list]

        for pred in self.all_used_predicate_names: # todo: iterate over base, goal, and neg goal seperately
            edge_type = (self.object_type_name, pred, ar)




        num_objects = kwargs.get("num_objects", None)
        if num_objects is None:
            num_objects = len([None for n in graph.nodes.data("node_type").values() if n == self.object_type_name])
        result[self.object_type_name].x = torch.zeros(())
        return result
