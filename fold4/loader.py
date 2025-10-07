# fold4/loader.py
import torch, json, numpy as np

class Fold4Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.get("emb_dim", 512)
        self.ff_dim = config.get("ff_dim", 2048)
        self.n_layers = config.get("n_layers", 12)
        self.n_heads = config.get("n_heads", 8)
        self.num_classes = len(config["label_map"])

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.ff_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ff_dim, self.num_classes)
        )

    def forward(self, x):
        return self.net(x)

def load_fold4_model(weight_path, config_path, device="cpu"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = Fold4Model(config)
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing: print("[Fold4] Missing keys:", missing)
    if unexpected: print("[Fold4] Unexpected keys:", unexpected)
    model.to(device).eval()

    id2label = {v: k for k, v in config["label_map"].items()}
    return model, id2label

def fold4_predict(model, ecg_signal, device="cpu"):
    x = np.asarray(ecg_signal, dtype=np.float32)
    if x.size < 512:
        x = np.pad(x, (0, 512 - x.size))
    else:
        x = x[:512]

    x = torch.tensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    idx = int(np.argmax(probs))
    return idx, {str(i): float(p) for i, p in enumerate(probs)}
