from typing import Any, Optional

import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing, Aggregation
from torch_geometric.typing import Adj


class AtomToObjectMP(MessagePassing):
    def __init__(
            self,
            embedding_size: int
    ):
        super().__init__(
            aggr=None
        )
        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pos: int, num_objects: int, **kwargs: Any) -> Any:  # dst_x: torch.Tensor
        r"""Runs the forward pass of the module."""
        n = x.shape[0]
        m = num_objects
        out = self.propagate(edge_index, x=x, size=(n, m), pos=pos, **kwargs)
        return out

    def message(self, x_j: Tensor, pos: int) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        # if we want the message to be 𝐱^(𝑖)_𝑜_𝑗 + (𝐦_𝑞)_𝑗, would need to add x_i to the result
        p = pos * self.embedding_size
        return x_j[:, p:p + self.embedding_size]

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        print(f"Input has shape {inputs.shape}. Correct? Should be [{self.embedding_size}]")
        return inputs

    # can we really skip aggregation? Wouldn't this result in tensors of inhomogeneous site?
    def aggregate(  # dont do anything
        self,
        inputs: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return inputs.view(-1, self.embedding_size)  # todo does this stack them correctly?