# RFC 0001: hybrid-search-service design

- **Status:** Accepted for M1
- **Author:** EquinoxWN
- **Created:** 2026-10-18

## Problem

Keyword search finds documents that contain the words of the query. People often describe what
they want in other words: "pause the program for a while" when the document says "Sleep pauses
the current goroutine". Vector search over sentence embeddings finds those, but misses exact
identifiers and rare terms that keyword search handles well. Hybrid search runs both and merges
the lists. Whether the merge actually helps has to be measured on judged queries, not assumed.

This project builds a hybrid search service on PostgreSQL (full-text search plus pgvector), so a
team that already runs Postgres needs no second search system, and measures each mode.

**Users:** backend engineers adding search to a product that already uses PostgreSQL, and people
learning how keyword, vector and hybrid retrieval compare.

## Define: critical-to-quality measures (CTQs)

| CTQ | Target | Measured by |
|---|---|---|
| Embeddings are correct | token IDs identical to Hugging Face `tokenizers`, cosine >= 0.99999 and max abs difference <= 1e-4 against the official ONNX export on 12 reference texts | `TestMatchesTheReferenceEncoder` |
| Every document indexed | table count equals corpus count after `hsearch index` | `scripts/e2e.sh` |
| Fusion is correct | RRF scores equal the sum of 1/(60 + rank); stable order; no duplicates; limit respected | `TestRRFProperties` (500 random cases), worked example |
| Hybrid is not worse than its parts | hybrid nDCG@10 >= the better single mode on the judged queries | `hsearch eval` |
| Queries are bounded and safe | queries 1..512 bytes, k 1..50, parameterised SQL, generic 500 errors | search and HTTP tests, store tests with a hostile query |
| Latency | single query under 200 ms on the laptop (including query embedding) | `hsearch query` timings |
| Reproducible | pinned model revision and SHA-256, pinned PostgreSQL and pgvector builds, corpus regenerated from the pinned Go toolchain | scripts, CI |

## Goals (M1)

- Documents are indexed for keyword search and embedded as vectors with pgvector.
- Each query runs both searches, and reciprocal rank fusion merges the two result lists.

## Non-goals

- Cross-encoder reranking (M2), the 200-query judged set (M2), a Meilisearch baseline (M3),
  facets, autocomplete and highlighting (M3).
- Ingest APIs, authentication, multi-tenancy, or running as a hosted service.
- Tuning the HNSW index beyond defaults; the corpus is small enough that recall loss is not the
  issue being studied.

## Design

```
hsearch corpus  Go 1.26.10 standard library docs -> 2,281 JSONL documents (id, title, body)
hsearch index   pure-Go MiniLM encoder (8 goroutines) -> 384-dim unit vectors
                -> PostgreSQL 18: documents(id, title, body,
                                            tsv tsvector GENERATED (title weight A, body weight B) + GIN,
                                            embedding vector(384) + HNSW vector_cosine_ops)
hsearch serve   GET /search?q&mode&k -> keyword: websearch_to_tsquery + ts_rank_cd (top 50)
                                     -> vector: embed query, ORDER BY embedding <=> q (top 50)
                                     -> hybrid: both in parallel, RRF (k = 60), top k
hsearch eval    35 hand-judged queries -> Recall@10, MRR@10, nDCG@10 per mode
```

- **Encoder** (`internal/embed`): sentence-transformers/all-MiniLM-L6-v2 (6 layers, 384 wide,
  Apache-2.0) implemented in Go: safetensors loader with bounds checks, uncased BERT tokenizer
  (cleaning, accent stripping, punctuation and CJK splitting, greedy WordPiece), the transformer
  layers with exact GELU, mean pooling and L2 normalisation. No cgo, no model server (ADR 0002).
  Inputs are cut at 256 tokens, as in the model's configuration.
- **Store** (`internal/store`): pgx pool; schema created by `Migrate`; batch upserts; all SQL is
  parameterised; the embedding size is checked and non-finite values refused;
  `hnsw.ef_search = 200` on every connection so a top-50 vector query is not cut short.
- **Fusion** (`internal/fusion`): reciprocal rank fusion, score = sum over lists of 1/(60 + rank);
  only ranks are used, so the incomparable scales of `ts_rank_cd` and cosine similarity do not matter.
- **Keyword list for hybrid**: all query terms are required (ADR 0003).
- **API** (`internal/search`): loopback by default, server timeouts, query and k limits,
  `400` with a reason for bad input, a generic `500` for backend errors, `/healthz` pings the database.
- **Corpus** (`internal/corpus`): exported, documented functions, types, methods and package docs
  of 46 standard-library packages, read with `go/doc` for linux/amd64 on every platform.

## Alternatives considered

| Option | Why not (yet) |
|---|---|
| Elasticsearch or OpenSearch (BM25 + kNN in one engine) | A second stateful system to run and keep in sync with the primary database. The point here is how far PostgreSQL alone goes; a dedicated engine (Meilisearch) is the M3 baseline. |
| Embeddings from an external API (OpenAI, Cohere) | Per-call cost, a network dependency and data leaving the machine on every query; results depend on a model version outside our control. |
| ONNX Runtime from Go via cgo, or a Python embedding sidecar | Fast, but brings a C library or a second service and runtime into every build and deployment. A pure-Go encoder is slower (about 16 documents per second on 8 goroutines) but has no native dependencies and is checked against ONNX Runtime (ADR 0002). |
| Weighted score sum (alpha x cosine + (1 - alpha) x normalised rank score) | Needs score normalisation and a tuned alpha per corpus; RRF uses ranks only and has one robust constant. Can be compared in M2 with the larger judged set. |
| OR-ed keyword terms for the hybrid keyword list | Measured: better keyword recall alone, worse hybrid (ADR 0003). |

## Measurement plan

- Encoder: compare with ONNX Runtime and Hugging Face `tokenizers` on 12 reference texts.
- `make e2e` on the author's laptop against PostgreSQL 18.6 + pgvector 0.8.7 built locally (no
  Docker), and in CI against the `pgvector/pgvector:0.8.7-pg18` container: index time, table count,
  Recall@10, MRR@10 and nDCG@10 per mode on 35 queries the author judged, per-query first relevant
  rank, single-query latency.
- Results, root causes and controls in `docs/results/m1.md` (Measure, Analyze, Improve, Control).

## Milestones

- **M1:** Documents are indexed for keyword search and embedded as vectors with pgvector. Each query runs both searches, and reciprocal rank fusion merges the two result lists.
- **M2:** A cross-encoder reranks the top 50 results. About 200 judged queries score every configuration with nDCG@10 and MRR.
- **M3:** Meilisearch is scored on the same set as a baseline. The API adds facets, autocomplete and highlighting.

## Risks and open questions

- 35 queries judged by the author are a small, biased sample: they are written in natural
  language, which favours vector search. Differences of a few points are within noise; M2's 200
  judged queries are needed for conclusions.
- The corpus is API documentation (short, technical); results may not carry over to long-form text.
- Embedding the corpus in pure Go takes minutes; larger corpora need batching across machines or
  a native runtime.
- HNSW is approximate; with 2,281 documents and `ef_search = 200` exact and approximate top-50
  lists are expected to agree, but this was not measured.
