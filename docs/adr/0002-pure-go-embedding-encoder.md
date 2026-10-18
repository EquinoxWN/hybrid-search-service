# ADR 0002: Embed text with a pure-Go encoder, checked against ONNX Runtime

- **Status:** Accepted

## Context

The service needs sentence embeddings for every document at index time and for every query at
search time. The usual options are an embedding API, a Python sidecar (sentence-transformers),
or ONNX Runtime called from Go through cgo. The service is otherwise a single static Go binary,
and the build machines have no C toolchain configured for cgo. The model,
all-MiniLM-L6-v2, is small: 6 transformer layers, 384 wide, 22.7 million parameters.

## Decision

Implement the encoder in Go: read `model.safetensors` and `vocab.txt`, reproduce the uncased BERT
tokenizer and the transformer forward pass, then mean pooling and L2 normalisation. Prove it
against an independent implementation: `scripts/reference/make_reference.py` runs Hugging Face
`tokenizers` and ONNX Runtime on the official ONNX export of the same revision, and the Go test
requires identical token IDs and a cosine of at least 0.99999 on 12 texts (accents, punctuation,
CJK, control characters, emoji, a text longer than 256 tokens).

## Consequences

- One static binary, no cgo, no second service; the model files are fetched at a pinned revision
  with SHA-256 checks and deleted after a run.
- Measured agreement: identical tokens on all 12 texts, lowest cosine 0.9999999.
- Slower than optimised runtimes: about 16 documents per second on 8 goroutines on the laptop
  (2,281 documents in about 2.5 minutes) and about 40 to 80 ms per query including the database.
  Fine for this corpus and for queries; a large corpus would need SIMD kernels, batching or a
  native runtime.
- Any other BERT-style model needs its own reference check; the loader refuses unexpected shapes,
  activations and missing tensors instead of guessing.
