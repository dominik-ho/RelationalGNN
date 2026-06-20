# send some deterministic messages, see if the order they are received is determined only by the order of edge_indices
from typing import Any

import torch
from torch import Tensor
from torch_geometric.typing import EdgeType

from gnn.atom_to_object_mp import AtomToObjectMP

class MyModule(torch.nn.Module):
    def __init__(self, embedding_size, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.atom_to_object_mp = AtomToObjectMP(self.embedding_size)

    def forward(self, x_dict: dict[str, Tensor], edge_index_dict: dict[EdgeType, Tensor]) -> torch.Tensor:
        for edge_type, edge_indices in edge_index_dict.items():
            atom_type, pos, _ = edge_type
            pos = int(pos)
            messages = self.atom_to_object_mp.forward(x_dict[atom_type], edge_indices, pos, x_dict["obj"].shape[0])

        return messages

def get_data(
        num_objects: int,
        num_predicates: int,
        max_ar: int,
        embedding_size: int,
        seed: int = 42,
):
    generator = torch.random.manual_seed(seed)
    predicates = [f"pred_{i}" for i in range(num_predicates)]
    predicate_arity_dict = {
        p: i  # torch.randint(1, max_ar + 1, (1,), generator=generator).item()
        for i, p in enumerate(predicates)
    }
    predicate_arity_dict[predicates[0]] = max_ar
    x_dict = {"obj": torch.zeros((num_objects, embedding_size))} | {
        p: torch.arange(0, ar * embedding_size)
        for p, ar in predicate_arity_dict.items()
    }
    return x_dict, edge_index_dict

def main():
    data = get_data(10, 4, 3, 5)

if __name__ == "__main__":
    main()