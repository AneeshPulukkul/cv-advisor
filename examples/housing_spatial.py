"""Geographic housing prices -> spatial GroupKFold over KMeans blocks.

Nearby houses in train+test inflate scores; blocks hold out whole regions.
Run:  python examples/housing_spatial.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from cv_advisor import CVAdvisor
from cv_advisor.spatial import make_spatial_blocks

rng = np.random.default_rng(11)
N = 400
lat = rng.uniform(32.5, 42.0, size=N)
lon = rng.uniform(-124.4, -114.1, size=N)
price = 150_000 + 8_000 * (lat - 32) - 3_000 * (lon + 124) + rng.normal(0, 25_000, size=N)
df = pd.DataFrame({"lat": lat, "lon": lon, "sqft": rng.integers(500, 3000, size=N), "price": price})

advisor = CVAdvisor("price")
rec = advisor.advise(df, "price")
print("splitter:", rec.recommended_splitter, rec.parameters)
print("patterns:", rec.detected_patterns)

X = df[["lat", "lon", "sqft"]].to_numpy()
y = df["price"].to_numpy()
splitter = advisor.get_splitter(df, rec)  # block-aware: no groups= needed
folds = list(splitter.split(X, y))
blocks = make_spatial_blocks(df, "lat", "lon", n_splits=rec.parameters["n_splits"])
for i, (tr, te) in enumerate(folds):
    # spatial-holdout guarantee: no test region appears in train
    assert not set(blocks[tr]) & set(blocks[te]), f"region leak in fold {i}"
print(f"OK: all {len(folds)} folds hold out whole regions")
scores = cross_val_score(Ridge(), X, y, cv=folds,
                         scoring="neg_root_mean_squared_error")
print("RMSE per fold:", np.round(-scores, 0))
print("\nCopy-pasteable snippet:\n" + rec.code_snippet)
