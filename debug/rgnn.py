# %%
from torch_geometric.data import Batch
import os
from pathlib import Path

import pymimir

from gnn.relational_gnn import MLPReadoutRGNN
from graph_encoder import GraphEncoder
# %%
data_path = Path(os.getcwd()).parent / "data"
d = pymimir.Domain(data_path / "domain.pddl")
p = pymimir.Problem(d, data_path / "test_clear_probBLOCKS-3-0.pddl", "grounded")
# %%
s = p.get_initial_state()
s
# %%
# help(pymimir.GroundAction)
a1 = pymimir.GroundAction.new(d.get_action("pickup"), [p.get_object("a")], p)
a1
# %%
s1 = a1.apply(s)
s1
# %%
s1
# %%
a2 = pymimir.GroundAction.new(d.get_action("stack"), [p.get_object("a"), p.get_object("b")], p)
a2
# %%
s2 = a2.apply(s1)
s2
# %%
encoder = GraphEncoder(d, "obj")
# %%
encoder.predicate_arity_dict
# %%

states = [s, s1, s2]
encoded = [encoder.encode(state, p.get_objects()) for state in states]
as_pyg = [encoder.to_pyg(encoded) for encoded in encoded]
batch = Batch.from_data_list(as_pyg)

print(batch)

gnn = MLPReadoutRGNN(
    encoder.predicate_arity_dict,
    encoder.object_type_name,
    num_layers=5,
    dynamic_num_layers=False,
    embedding_size=32,
    activation="relu",
    aggregation="sum",
    manipulate_embeddings=None
)

gnn.forward(batch.x_dict, batch.edge_index_dict, batch.batch_dict)