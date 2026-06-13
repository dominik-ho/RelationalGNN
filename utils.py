from torch_geometric.typing import EdgeType


def invert_edge_type(edge_type: EdgeType) -> EdgeType:
    return edge_type[2], edge_type[1], edge_type[0]
