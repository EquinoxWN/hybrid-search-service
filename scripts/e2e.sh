#!/usr/bin/env bash
# End-to-end run against PostgreSQL + pgvector: build the corpus, embed and index it, score the
# three modes on the judged queries, then serve the API and query it over HTTP.
# Needs HSEARCH_DSN and the model (scripts/fetch-model.sh). Prints ::notice lines (::error on failure).
# Usage: scripts/e2e.sh <hsearch binary>
set -euo pipefail
BIN=${1:?usage: scripts/e2e.sh <hsearch binary>}
OUT=.tmp/e2e
mkdir -p "$OUT"
fail() { echo "::error title=hsearch e2e::$1"; exit 1; }
[ -n "${HSEARCH_DSN:-}" ] || fail "HSEARCH_DSN is not set"

"$BIN" corpus -goroot "$(go env GOROOT)" -out "$OUT/corpus.jsonl" | tee "$OUT/corpus.log"
docs=$(wc -l < "$OUT/corpus.jsonl" | tr -d ' ')

"$BIN" index -corpus "$OUT/corpus.jsonl" | tee "$OUT/index.log"
grep -q "table holds $docs\$" "$OUT/index.log" || fail "the table does not hold all $docs documents"
echo "::notice title=hsearch index::$(cat "$OUT/index.log")"

"$BIN" eval -corpus "$OUT/corpus.jsonl" | tee "$OUT/eval.log"
for mode in keyword vector hybrid; do
  grep -q "^$mode " "$OUT/eval.log" || fail "no $mode row in the evaluation"
  echo "::notice title=hsearch eval $mode::$(grep "^$mode " "$OUT/eval.log")"
done

for mode in keyword vector hybrid; do
  "$BIN" query -mode "$mode" -k 5 "create a temporary directory" | tee -a "$OUT/query.log"
done

# The API on a free loopback port, stopped at the end.
port=$((20000 + RANDOM % 20000))
"$BIN" serve -addr "127.0.0.1:$port" > "$OUT/serve.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -fs "http://127.0.0.1:$port/healthz" >/dev/null && break; sleep 0.2; done
curl -fs "http://127.0.0.1:$port/healthz" | tee "$OUT/api.log"
curl -fs "http://127.0.0.1:$port/search?q=create+a+temporary+directory&k=5" | tee -a "$OUT/api.log"
echo
grep -q '"id":"os.MkdirTemp"' "$OUT/api.log" || fail "the API did not return os.MkdirTemp in the top 5 for the temporary-directory query"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/search?q=&k=3")
[ "$code" = 400 ] || fail "empty query answered $code, want 400"
echo "::notice title=hsearch api::GET /search returned os.MkdirTemp in the top 5; empty query rejected with 400"
echo "E2E PASS"
