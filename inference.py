# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data import comp2graph, collate_batch
from model import CompositionEmbedding


# %%
model = torch.load("checkpoints/test.pt")
model

# %%
benchmark = pd.read_csv(
    "https://raw.githubusercontent.com/usccolumbia/cspbenchmark/main/data/CSPbenchmark_test_data.csv"
)
benchmark_ids = benchmark["material_id"]


def compute_scores(comp_1: str, other_comps: list[str], batch_size: int = 32):
    X1 = collate_batch([comp2graph(comp_1)])
    X2 = []
    for i in range(0, len(other_comps), batch_size):
        X2.append(
            collate_batch([comp2graph(c) for c in other_comps.iloc[i : i + batch_size]])
        )

    model.eval()
    z1 = model.embed(X1)
    z2 = torch.cat([model.embed(x) for x in X2])

    probs = model.to_probability(torch.cdist(z1, z2).reshape(-1))
    return probs.numpy(force=True)


# %%
scores = compute_scores("Nb3Si", benchmark["full_formula"])
scores.round(2)
