from collections import defaultdict
from functools import cache

import torch
from networkx import MultiDiGraph
import pymimir
from pymimir import State, Object, Domain, GroundAtom, GroundLiteral
from torch_geometric.data import HeteroData

import utils
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
        self.predicate_arity_dict = {   # don't encode 0-ary predicates and the ignored ones
            pred: ar
            for pred, ar in self.predicate_arity_dict.items()
            if ar > 0 and pred not in self.ignored_predicates
        }
        # self.all_used_predicate_names = set(self.predicate_arity_dict.keys())
        self.pos_goal_predicate_names_dict: dict[str, str] = dict()
        # self.pos_goal_predicate_to_base = {
        #     v: k for k, v in self.pos_goal_predicate_names_dict.items()
        # }
        self.neg_goal_predicate_names_dict: dict[str, str] = dict()
        self.extend_predicate_arity_dicts()
        # self.neg_goal_predicate_to_base = {
        #     v: k for k, v in self.neg_goal_predicate_names_dict.items()
        # }
        # self.all_used_predicate_names |= set(self.pos_goal_predicate_names_dict.values()) | set(self.neg_goal_predicate_names_dict.values())
        # self.full_predicate_arity_dict = self.predicate_arity_dict | {
        #     gpred: self.predicate_arity_dict[bpred]
        #     for gpred, bpred in self.pos_goal_predicate_to_base.items() | self.neg_goal_predicate_to_base.items()
        # }
        # print(self.predicate_arity_dict)
        self.edge_types = [
            (self.object_type_name, str(pos), pred)
            for pred, ar in self.predicate_arity_dict.items()
            for pos in range(ar)
        ]
        self.edge_types += [utils.invert_edge_type(e) for e in self.edge_types]


    # def get_arity_of(self, pred):
    #     ar = self.predicate_arity_dict.get(pred)
    #     if ar is None:
    #         base_pred = self.pos_goal_predicate_to_base.get(pred) or self.neg_goal_predicate_to_base.get(pred)
    #         assert base_pred is not None, f"predicate {pred} is not defined, {base_pred} not found!"
    #         ar = self.predicate_arity_dict[base_pred]
    #     return ar

    def extend_predicate_arity_dicts(self):
        pos_goal_predicate_arity_dict = dict()
        neg_goal_predicate_arity_dict = dict()
        for pred, ar in self.predicate_arity_dict.items():
            gpred = self.add_goal_postfix(pred)
            self.pos_goal_predicate_names_dict[pred] = gpred
            pos_goal_predicate_arity_dict[gpred] = ar

            ngpred = self.add_neg_prefix(gpred)
            self.neg_goal_predicate_names_dict[pred] = ngpred
            neg_goal_predicate_arity_dict[ngpred] = ar
        self.predicate_arity_dict.update(pos_goal_predicate_arity_dict | neg_goal_predicate_arity_dict)


    def encode(self, state: State, objects: list[Object]) -> MultiDiGraph:
        result = MultiDiGraph()
        result.add_nodes_from(objects, node_type=self.object_type_name)  # [o.get_name() for o in objects]
        for atom in state.get_atoms():
            if atom.get_predicate().get_name() in GraphEncoder.ignored_predicates or atom.get_predicate().get_arity() < 1:
                continue
            result.add_node(atom, node_type=atom.get_predicate().get_name(), class_type=GroundAtom)
            for i, o in enumerate(atom.get_terms()):
                # result.add_edges_from(((o, atom), (atom, o)), pos=i)
                result.add_edge(o, atom, pos=i)
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
                # result.add_edges_from(((o, goal_literal), (goal_literal, o)), pos=i)
                result.add_edge(o, goal_literal, pos=i)
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
        # self.all_used_predicate_names.add(result)
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
        # self.all_used_predicate_names.add(result)
        return result

    @property
    def all_used_predicate_names(self):
        return self.predicate_arity_dict.keys() | self.pos_goal_predicate_names_dict.keys() | self.neg_goal_predicate_names_dict.keys()

    def to_pyg(self, graph: MultiDiGraph, **kwargs) -> HeteroData:
        result = HeteroData()

        objects_list: list[Object] = []
        pyg_atom_nodes = {
            p: [] for p in self.all_used_predicate_names
        }
        atoms_index_dict: dict[GroundAtom | GroundLiteral, int] = dict()
        num_atoms_dict = defaultdict(int)
        for node, data in graph.nodes.items():
            node_type = data["node_type"]
            if node_type == self.object_type_name:
                assert isinstance(node, Object), "Object node is not an object!"
                objects_list.append(node)
                continue
            assert isinstance(node, GroundAtom | GroundLiteral), "GroundAtom | GroundLiteral node is not an atom or literal!"
            pyg_atom_nodes[node_type].append(node)
            n = num_atoms_dict[node_type]
            atoms_index_dict[node] = n
            num_atoms_dict[node_type] = n + 1

        objects_list.sort(key=lambda x: x.get_name())
        objects_index_dict = {
            o: i for i, o in enumerate(objects_list)
        }

        result[self.object_type_name].x = torch.zeros(len(objects_list))
        for pred, nodes in pyg_atom_nodes.items():
            ar = self.predicate_arity_dict[pred]
            result[pred].x = torch.zeros((len(nodes), ar))
        result.obj_names = [o.get_name() for o in objects_list]

        # for pred, ar in self.predicate_arity_dict.items(): # todo: iterate over base, goal, and neg goal seperately
        #     # ar = self.predicate_arity_dict[pred]
        #     for pos in range(ar):
        #         edge_type = (self.object_type_name, pred, pos)
        #         data[edge_type].x = torch.zeros()

        # edges_list_dict = defaultdict(lambda: (list(), list()))
        edges_list_dict = {
            edge_type: (list(), list())
            for edge_type in self.edge_types
        }
        for src, dst, pos in graph.edges(data="pos"):
            # src, dst, pos = edge   #todo extract pos
            spos = str(pos)
            if isinstance(src, Object):
                dst: GroundAtom | GroundLiteral
                atom_node_type = graph.nodes[dst]["node_type"]
                edge_type = (self.object_type_name, spos, atom_node_type)

                src_index = objects_index_dict[src]
                dst_index = atoms_index_dict[dst]

                edges_list_dict[edge_type][0].append(src_index)
                edges_list_dict[edge_type][1].append(dst_index)

                inverted = utils.invert_edge_type(edge_type)
                edges_list_dict[inverted][0].append(dst_index)
                edges_list_dict[inverted][1].append(src_index)
        for edge_type, (src_indices, dst_indices) in edges_list_dict.items():
            result[edge_type].edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)


        # num_objects = kwargs.get("num_objects", None)
        # if num_objects is None:
        #     num_objects = len([None for n in graph.nodes.data("node_type").values() if n == self.object_type_name])

        return result
