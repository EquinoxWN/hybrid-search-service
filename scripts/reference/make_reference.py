"""Write testdata/minilm-reference.json from the official all-MiniLM-L6-v2 ONNX export.

The Go encoder is tested against this file: token IDs from the Hugging Face `tokenizers` library
and sentence embeddings (mean pooling, L2 normalised) computed by ONNX Runtime. Neither shares
code with the Go implementation.

usage: python scripts/reference/make_reference.py <model dir> <output json>
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_LEN = 256

CASES = [
    "",
    "convert a string to an integer",
    "How do I read a file line by line?",
    "strconv.Atoi converts a decimal string to an int.",
    "Café crème brûlée, naïve façade, Ångström and Øresund",
    "HTTP/2 server push: GET /api/v1/items?id=42&sort=desc",
    "time.Sleep pauses the current goroutine for at least the duration d.",
    "\tTabs,\nnewlines\r\nand   repeated    spaces everywhere",
    "数据库 索引 与 向量 搜索",
    "emoji 🚀 and symbols ∑ ≤ ≥ → are rare",
    "UPPER lower MiXeD case WordPiece unaffordable antidisestablishmentarianism",
    " ".join(["the quick brown fox jumps over the lazy dog"] * 40),
]


def main() -> None:
    """Tokenize and embed every case with the reference stack."""
    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_LEN)
    tok.no_padding()
    sess = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])
    cases = []
    for text in CASES:
        enc = tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
        hidden = sess.run(None, feeds)[0][0]
        pooled = hidden.mean(axis=0)
        pooled = pooled / max(float(np.linalg.norm(pooled)), 1e-12)
        cases.append({"text": text, "ids": enc.ids, "embedding": [round(float(x), 7) for x in pooled]})
    doc = {"model": "sentence-transformers/all-MiniLM-L6-v2", "revision": REVISION, "max_length": MAX_LEN, "cases": cases}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
