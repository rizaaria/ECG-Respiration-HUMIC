import torch, json, numpy as np

class SimpleConformer(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.input_proj = torch.nn.Linear(1, 64)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(64, num_classes)
    def forward(self, x):
        # x: (B, 512) → (B, 512, 1)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x = self.relu(self.input_proj(x))
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

def load_conformer_model(weight_path, config_path, device="cpu"):
    with open(config_path) as f:
        cfg = json.load(f)
    classes = cfg.get("class_names", ["L","N","Q","R","V"])
    model = SimpleConformer(num_classes=len(classes))
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[Conformer] missing:", missing)
    print("[Conformer] unexpected:", unexpected)
    model.to(device).eval()
    return model, classes

def conformer_predict(model, ecg_signal, device="cpu"):
    x = np.asarray(ecg_signal, dtype=np.float32)
    if x.size < 512: x = np.pad(x, (0, 512 - x.size))
    else: x = x[:512]
    x = torch.tensor(x).unsqueeze(0).to(device)  # (1,512)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    idx = int(np.argmax(probs))
    return idx, {str(i): float(p) for i, p in enumerate(probs)}
