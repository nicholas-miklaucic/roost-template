import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

# df =
# pd.read_pickle('https://github.com/Minoru938/CSPML/raw/refs/heads/master/CSPML_latest_codes/data_set/MP_stable_20211107.pd.xz')
# name = ''
df = pd.read_feather("data/cspbench/data.feather")
name = "data/cspbench/"


# ----------------------------------------


common_ratios = (
    df.query('total_atoms <= 20 and ratio_class != "others"')["ratio_class"]
    .value_counts()
    .head(20)
)
common_ratios

df_ratio = df.query('comp_ratio_label == "3:1:1"')
subset = df_ratio.sample(100, random_state=43)

Y = np.array(list(subset["strfgp"]))


rng = np.random.default_rng(1234)
pdist_batch_size = 0

dataset = []
for ratio in tqdm(common_ratios.index[::-1]):
    df_ratio: pd.DataFrame = df.query("comp_ratio_label == @ratio").reset_index()

    if pdist_batch_size == 0 or pdist_batch_size >= len(df_ratio.index):
        groups = [np.arange(len(df_ratio.index))]
    else:
        groups = np.array_split(rng.permutation(df_ratio.shape[0]), pdist_batch_size)
    for group in groups:
        subset = df_ratio.iloc[group]
        Y = np.array(list(subset["strfgp"]))
        dists = pdist(Y, "euclidean")
        dm = squareform(dists)
        ii, jj = np.triu_indices_from(dm, k=1)
        cols = ["id", "full_formula", "space_group"]
        data_dict = {"ratio": [ratio for _ in ii], "dist": dm[(ii, jj)]}
        for inds, suff in zip((ii, jj), ("_1", "_2")):
            for col in cols:
                data_dict[col + suff] = subset.iloc[inds][col].values

        dataset.append(pd.DataFrame(data_dict))

data_df = pd.concat(dataset).reset_index(drop=True)
for col in (
    "ratio",
    "id_1",
    "full_formula_1",
    "space_group_1",
    "id_2",
    "full_formula_2",
    "space_group_2",
):
    data_df[col] = data_df[col].astype("category")

data_df.to_feather(f"{name}pairs_data.feather")

print("Done!")
