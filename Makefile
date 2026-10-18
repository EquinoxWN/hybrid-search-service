.PHONY: setup lint test audit build model clean-model reference pg-setup pg-start pg-stop e2e bench ci

EXE := $(if $(filter Windows_NT,$(OS)),.exe,)
HSEARCH := .tmp/bin/hsearch$(EXE)
REFPY := $(if $(filter Windows_NT,$(OS)),.tmp/refvenv/Scripts/python,.tmp/refvenv/bin/python)
SRC := cmd internal
# CI adds -race (it needs cgo, which not every machine has).
GOTESTFLAGS ?=

setup:
	go mod download
	go mod verify

lint:
	@files="$$(gofmt -l $(SRC))"; test -z "$$files" || (echo "$$files"; echo "run gofmt -w $(SRC)"; exit 1)
	go vet ./...

# Unit tests always; the encoder reference test needs `make model`, the PostgreSQL tests need
# HSEARCH_TEST_DSN (a database the tests may recreate the documents table in).
test:
	go test $(GOTESTFLAGS) -count=1 ./...

# Known vulnerabilities in the Go code paths actually called (Go 1.26.9, as in CI).
audit:
	go run golang.org/x/vuln/cmd/govulncheck@v1.8.0 ./...

build:
	go build -trimpath -o $(HSEARCH) ./cmd/hsearch

# all-MiniLM-L6-v2 at a pinned revision, SHA-256 checked, into .tmp/models.
model:
	bash scripts/fetch-model.sh

# Delete the model after a run (the results doc records how to fetch it again).
clean-model:
	rm -rf .tmp/models

# Regenerate testdata/minilm-reference.json with Hugging Face tokenizers and ONNX Runtime.
reference:
	bash scripts/fetch-model.sh --reference
	python -m venv .tmp/refvenv
	$(REFPY) -m pip install -q -r scripts/reference/requirements.txt
	$(REFPY) -I scripts/reference/make_reference.py .tmp/models/all-MiniLM-L6-v2 testdata/minilm-reference.json

# Windows without Docker: PostgreSQL 18 + pgvector built into .tmp (see scripts/pg-local.sh).
pg-setup:
	bash scripts/pg-local.sh setup
pg-start:
	bash scripts/pg-local.sh start
pg-stop:
	bash scripts/pg-local.sh stop

# Corpus, index, evaluation and API against HSEARCH_DSN; the numbers in docs/results/m1.md.
e2e: build model
	bash scripts/e2e.sh $(HSEARCH)

bench: e2e

ci: setup lint test
