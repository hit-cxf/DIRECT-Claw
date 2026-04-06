import numpy as np

def get_energy_score(v: float, v0: float) -> float:
    """
    Args
    v: expected energy
    v0: actual energy
    """
    log_v = np.log(v + 1e-8)
    log_v0 = np.log(v0 + 1e-8)
    return np.exp(-min(log_v - log_v0, (v - v0) / 40) ** 2 / 2)