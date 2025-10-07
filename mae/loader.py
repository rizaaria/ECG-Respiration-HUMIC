# app/models/MAE/loader.py
from __future__ import annotations
from pathlib import Path
import json
import torch
import numpy as np
from .mae_builder import TransformerMAE

class MAERunner:
    def __init__(self, model: TransformerMAE, device: str = "cpu", idx_to_class: dict[int, str] | None = None):
        self.model = model.to(device).eval()
        self.device = device
        self.idx_to_class = idx_to_class or {0: "N", 1: "R", 2: "Q", 3: "V", 4: "L"}

    @torch.no_grad()
    def predict(self, segments_2d):
        """
        segments_2d: np.ndarray (B, T) float32
        return: (pred_ids: np.ndarray[B], logits: torch.Tensor[B,C])
        """
        if isinstance(segments_2d, np.ndarray):
            x = torch.from_numpy(segments_2d).float()
        else:
            x = segments_2d.float()
        x = x.unsqueeze(1).to(self.device)  # (B,1,T)

        logits, _ = self.model(x)           # (B,C)
        pred = torch.argmax(logits, dim=1)  # (B,)
        return pred.cpu().numpy(), logits

def _read_json(p: Path, default=None):
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default

def load_mae(weights_dir: str, device: str = "cpu"):
    """
    Mengharapkan file di dalam weights_dir:
      - *.pth                 → bobot model
      - class_mapping.json    → idx_to_class / class_to_idx (urutan kelas training)
      - config_hpo.json       → hyperparam arsitektur (patch_size, dim, n_heads, n_layers, mask_ratio, dropout)
    Return: (runner, label_map, id2label)
    """
    wdir = Path(weights_dir)

    # cari file .pth
    ckpt = None
    for p in sorted(wdir.glob("*.pth")):
        ckpt = p
        break
    if ckpt is None:
        raise FileNotFoundError(f"Tidak menemukan file .pth di {weights_dir}")

    cfg = _read_json(wdir / "config_hpo.json", default={})
    mdl_cfg = cfg.get("model", {})
    dim        = int(mdl_cfg.get("dim", 128))
    n_heads    = int(mdl_cfg.get("n_heads", 4))
    n_layers   = int(mdl_cfg.get("n_layers", 6))
    patch_size = int(mdl_cfg.get("patch_size", 16))
    mask_ratio = float(mdl_cfg.get("mask_ratio", 0.5))
    dropout    = float(mdl_cfg.get("dropout", 0.1))

    class_map = _read_json(wdir / "class_mapping.json", default=None)
    if class_map and "idx_to_class" in class_map:
        idx_to_class = {int(k): v for k, v in class_map["idx_to_class"].items()}
        class_to_idx = {k: int(v) for k, v in class_map["class_to_idx"].items()}
    else:
        idx_to_class = {0: "N", 1: "R", 2: "Q", 3: "V", 4: "L"}
        class_to_idx = {v: k for k, v in idx_to_class.items()}

    # bangun model kosong
    model = TransformerMAE(
        dim=dim, n_heads=n_heads, n_layers=n_layers,
        patch_size=patch_size, mask_ratio=mask_ratio, dropout=dropout,
        n_classes=len(idx_to_class)
    )

    # muat state_dict
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # bersihkan prefix "module."
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    # --- handle pos_embed size mismatch ---
    if "pos_embed" in new_state:
        pe = new_state["pos_embed"]
        if tuple(pe.shape) != tuple(model.pos_embed.shape):
            # Samakan bentuk parameter model dengan yang ada di checkpoint,
            # supaya load_state_dict tidak error. (Forward tetap resize dinamis.)
            with torch.no_grad():
                model.pos_embed = torch.nn.Parameter(torch.zeros_like(pe))

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[MAE] missing keys: {missing}")
    if unexpected:
        print(f"[MAE] unexpected keys: {unexpected}")

    runner = MAERunner(model=model, device=device, idx_to_class=idx_to_class)
    id2label = idx_to_class
    label_map = class_to_idx
    return runner, label_map, id2label
