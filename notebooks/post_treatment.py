# %%
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from path import Path

# %%
basedir = Path("../data/sims/one_sided/three_eq/")
outdir = Path("../data/sims/one_sided/three_eq_last/").mkdir_p()
for file in basedir.files():
    df = pd.read_csv(file)
    df = df[df.t == df.t.max()].set_index("x").drop("t", axis=1)
    df.to_csv(outdir / file.basename())


# %%
ds = df[~df.index.duplicated()].to_xarray()
# %%
dsl = ds.isel(t=-1)

# %%
fig, ax = plt.subplots(figsize=(10, 3))
dsl["qₚ"].plot(ax=ax)
# ax.set_xlim(0, 100.0)
# ax.set_ylim(0, 1.8)
# %%
