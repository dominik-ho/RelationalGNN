from functools import cache

import torch
from networkx import MultiDiGraph
import pymimir
from pymimir import State, Object, Domain
from torch_geometric.data import HeteroData


class GraphEncoder:
    def __init__(
            self,
            # predicate_arity_dict: dict[str, int],
            domain: Domain,
            object_type_name: str
    ):
        # self.predicate_arity_dict = predicate_arity_dict
        self.domain = domain
        self.object_type_name = object_type_name
        self.used_predicate_names = {pred.get_name() for pred in self.domain.get_predicates()}

    def encode(self, state: State, objects: list[Object]) -> MultiDiGraph:
        result = MultiDiGraph()
        result.add_nodes_from(objects, type=self.object_type_name)  # [o.get_name() for o in objects]
        for atom in state.get_atoms():
            if atom.get_predicate().get_name() == "object":
                continue
            result.add_node(atom, type=atom.get_predicate().get_name())
            for i, o in enumerate(atom.get_terms()):
                result.add_edges_from(((o, atom), (atom, o)), pos=i)
        for atom in state.get_problem().get_goal_condition().get_literals().get_atom():
            if atom.get_predicate().get_name() == "object":
                continue
            result.add_node(atom, type=self.add_neg_prefix(self.add_goal_postfix(atom.get_predicate().get_name())))
            for i, o in enumerate(atom.get_terms()):
                result.add_edges_from(((o, atom), (atom, o)), pos=i)
        return result

    @cache
    def add_goal_postfix(self, predicate_name: str):
        postfix = "_G"
        result = predicate_name + postfix
        while predicate_name in self.used_predicate_names:
            result += postfix
        self.used_predicate_names.add(result)
        return result

    @cache
    def add_neg_prefix(self, predicate_name: str):
        prefix = "not:"
        result = prefix + predicate_name
        while predicate_name in self.used_predicate_names:
            result = result + prefix
        self.used_predicate_names.add(result)
        return result

    def to_pyg(self, graph: MultiDiGraph, **kwargs) -> HeteroData:
        result = HeteroData()

        num_objects = kwargs.get("num_objects", None)
        if num_objects is None:
            num_objects = len([None for n in graph.nodes.data("type").values() if n == self.object_type_name])
        result[self.object_type_name].x = torch.zeros(())
        return result
