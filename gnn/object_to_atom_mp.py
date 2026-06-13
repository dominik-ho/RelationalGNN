from typing import Any

import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj


class ObjectToAtomMP(MessagePassing):
    def __init__(
            self,
            embedding_size: int,
    ):
        super().__init__(
            aggr="sum"  # aggr should not matter, as exactly 1 message per pos will be received by an atom node
        )
        self.embedding_size = embedding_size

    def forward(self, src_x: torch.Tensor, edge_index: torch.Tensor, **kwargs: Any) -> Any:  # dst_x: torch.Tensor
        r"""Runs the forward pass of the module."""
        out = self.propagate(edge_index, src_x=src_x, **kwargs)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        print(f"Input has shape {inputs.shape}. Correct? Should be [{self.embedding_size}]")
        return inputs
