from pathlib import Path
from transformers import AutoTokenizer, BertForSequenceClassification
import torch, json

class BertBeatModel:
    """
    Loader + predictor untuk BERT yang dilatih dengan representasi teks angka (0..255).
    Mengembalikan label string 'N','L','R','V','Q'.
    """
    def __init__(self, weights_dir: str):
        p = Path(weights_dir)
        if not p.exists():
            raise FileNotFoundError(f"weights_dir tidak ditemukan: {p}")

        self.tokenizer = AutoTokenizer.from_pretrained(p)
        self.model = BertForSequenceClassification.from_pretrained(p)
        self.model.eval()

        # Mapping label: ambil dari config bila ada; jika masih LABEL_0.., override dari label_map.json
        cfg = self.model.config
        self.id2label = None

        try:
            # Coba mapping dari config
            if getattr(cfg, "id2label", None) and cfg.id2label:
                # id2label di config bisa ber-key string; normalisasi ke int
                tmp = {int(k): v for k, v in cfg.id2label.items()}
                # jika masih generik "LABEL_0", kita override
                if set(tmp.values()) == {f"LABEL_{i}" for i in range(len(tmp))}:
                    raise KeyError("generic labels")
                self.id2label = tmp
        except Exception:
            pass

        if self.id2label is None:
            # Override dari label_map.json
            lm_path = p / "label_map.json"
            if not lm_path.exists():
                raise FileNotFoundError("label_map.json tidak ditemukan, dan config tidak punya id2label yang benar.")
            lm = json.loads(lm_path.read_text())
            # lm: {"N":0,"L":1,"R":2,"V":3,"Q":4}
            label2id = {k:int(v) for k,v in lm.items()}
            id2label = {v:k for k,v in label2id.items()}
            cfg.label2id = label2id
            cfg.id2label = id2label
            cfg.num_labels = len(label2id)
            self.id2label = id2label

    @torch.no_grad()
    def predict_texts(self, texts: list[str]):
        enc = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=512, return_tensors="pt"
        )
        outputs = self.model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)
        top = torch.argmax(probs, dim=-1)
        res = []
        for i in range(len(texts)):
            res.append({
                "label": self.id2label[int(top[i])],
                "proba": probs[i].tolist()
            })
        return res
