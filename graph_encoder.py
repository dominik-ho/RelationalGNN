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

    def encode(self, state: State, objects: list[Object]) -> MultiDiGraph:
        result = MultiDiGraph()
        result.add_nodes_from(objects, type=self.object_type_name)  # [o.get_name() for o in objects]

        for atom in state.get_atoms():
            if atom.get_predicate().get_name() == "object":
                continue
            result.add_node(atom, type=atom.get_predicate())
            for i, o in enumerate(atom.get_terms()):
                # result.add_edge(o, atom, pos=i)
                # result.add_edge(atom, o, pos=i)
                result.add_edges_from(((o, atom), (atom, o)), pos=i)
        return result

    def to_pyg(self, graph: MultiDiGraph) -> HeteroData:
        pass
