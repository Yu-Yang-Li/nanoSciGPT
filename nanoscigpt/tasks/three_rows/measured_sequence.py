"""Train the three regressors in native measurement units, using shared supervised Nano settings."""
import torch
from nanoscigpt.tasks.three_rows import paired_numeric

def fit(seed, settings, train, aux, tests):
    unit = settings["unit"]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        t = {"x": train["x"].double(), "y": train["y"].double()/unit}
        u = {k:v.double() for k,v in aux.items()}
        v = [{"x": test["x"].double()} for test in tests]
        return [[p*unit for p in arm] for arm in paired_numeric.fit(seed,settings,t,u,v)]
    finally:
        torch.set_default_dtype(previous)
