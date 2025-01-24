import numpy as np
import pandas as pd
from pymatgen.core import Structure, Composition
from pathlib import Path
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from math import gcd
import time


data_folder = Path("data/cspbench")

# ----------------------------------------

structs = {}
for fn in data_folder.glob("*.cif"):
    structs[fn.stem] = Structure.from_file(fn)

# https://github.com/Minoru938/CSPML/blob/master/CSPML_latest_codes/Create_strcmp_fgp.ipynb
structures = structs.values()

# Site featurizer.
cnnf = CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0)


def parallel_cnnf(featurizer, str_x):
    return np.array([featurizer(str_x, i) for i in range(len(str_x.sites))])


# SiteStats.
def SiteStats(site_fgps):
    return np.array(
        [site_fgps.mean(0), site_fgps.std(0), site_fgps.min(0), site_fgps.max(0)]
    ).T.flatten()


# Calculate structure fingerprints for all stable data.
n_iter = len(structures)

strfgp_stable = []
errors_i = []

s = time.time()

for i, str_x in enumerate(structures):
    strfgp_stable.append(
        SiteStats(parallel_cnnf(cnnf.featurize, str_x))
    )  # site fgps for the ith str.

e = time.time()
# print(f"time: {e-s}")
# print(f"time per iteration: {(e-s)/n_iter}")

# Save results.
strfgp_stable_array = np.array(strfgp_stable)

# print(strfgp_stable_array.shape)

np.save(data_folder / "sitefgps", strfgp_stable_array)

fdf = pd.DataFrame(
    index=structs.keys(),
    data={
        "struct": structs.values(),
        "strfgp": strfgp_stable,
    },
)
fdf.index.name = "id"


def reduce(ratio):
    div = gcd(*ratio)
    return [r // div for r in ratio]


fdf["comp"] = [s.composition for s in fdf["struct"]]
fdf["full_formula"] = [c.to_pretty_string() for c in fdf["comp"]]
fdf["total_atoms"] = [int(c.num_atoms) for c in fdf["comp"]]
fdf["ratio_class"] = [
    ":".join(
        [str(x) for x in reduce(sorted(map(int, c.get_el_amt_dict().values())))[::-1]]
    )
    for c in fdf["comp"]
]
fdf.reset_index().drop(columns=["struct", "comp"]).to_feather(
    data_folder / "data.feather"
)

print("Done!")
