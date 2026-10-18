# Contributing

Thanks for helping. This is a Go project (PostgreSQL with pgvector); issues and pull requests are
welcome.

## Set up and check your change

Needs Go 1.26+, make, bash with curl, and PostgreSQL 18 with pgvector 0.8.7 (`make pg-setup` and
`make pg-start` build one into `.tmp/` on Windows; elsewhere use the `pgvector/pgvector:0.8.7-pg18`
image). Downloads, the model and build output stay in git-ignored `.tmp/`.

```bash
make setup         # go mod download and verify
make model         # pinned all-MiniLM-L6-v2, needed by the encoder reference test
make lint          # gofmt, go vet
HSEARCH_TEST_DSN="postgres://postgres@127.0.0.1:54329/hsearch_test?sslmode=disable" make test
make audit         # govulncheck
make e2e           # needs HSEARCH_DSN: corpus, index, eval, API
make clean-model   # delete the model when you are done
```

A pull request is ready when `make lint` and `make test` pass (with the model and a test
database, so nothing is skipped) and CI is green.

## Guidelines

- Changes to the encoder must keep `TestMatchesTheReferenceEncoder` passing; regenerate
  `testdata/minilm-reference.json` with `make reference` only when the model revision changes.
- Changes to ranking must come with `hsearch eval -per-query` before and after, in the pull request.
- Judged queries need relevant documents that exist in the corpus; `hsearch eval` refuses otherwise.
- Add or update a test for every behaviour change, and update `docs/results/` when numbers change.
- Significant design changes need an ADR in `docs/adr/` (copy `0001` for the format).
- Keep function comments to one short phrase; let names and tests explain the rest.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your work is released under the [MIT License](LICENSE).
