from typing import Iterable

from pymimir import Predicate
from torch_geometric.typing import EdgeType


def invert_edge_type(edge_type: EdgeType) -> EdgeType:
    return edge_type[2], edge_type[1], edge_type[0]

def get_predicate_arity_dict(predicates: Iterable[Predicate]) -> dict[str, int]:
    return {
        predicate.get_name(): predicate.get_arity()
        for predicate in predicates
    }
