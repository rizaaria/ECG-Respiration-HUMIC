import numpy as np
def signal_to_text(sig: np.ndarray) -> str:
    sig = sig.astype(np.float32)
    ptp = sig.max() - sig.min()
    norm = ((sig - sig.min()) / (ptp + 1e-8) * 255.0).astype(np.int32)
    return " ".join(map(str, norm.tolist()))