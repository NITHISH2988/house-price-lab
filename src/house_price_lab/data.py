import numpy as np
import pandas as pd


def make_dataset(n=1200, seed=42):
    r = np.random.default_rng(seed)
    area = r.integers(450, 3200, n)
    beds = np.clip((area / 520 + r.normal(0, 0.7, n)).round(), 1, 6)
    age = r.integers(0, 55, n)
    loc = r.choice(["central", "suburban", "rural"], n)
    kind = r.choice(["apartment", "house", "townhouse"], n)
    price = (
        45000
        + area * 165
        + beds * 9000
        - age * 1100
        + pd.Series(loc)
        .map({"central": 75000, "suburban": 32000, "rural": 0})
        .to_numpy()
        + pd.Series(kind)
        .map({"apartment": 0, "house": 22000, "townhouse": 12000})
        .to_numpy()
        + r.normal(0, 28000, n)
    )
    return pd.DataFrame(
        {
            "area_sqft": area,
            "bedrooms": beds,
            "age_years": age,
            "location": loc,
            "property_type": kind,
            "price": price,
        }
    )
