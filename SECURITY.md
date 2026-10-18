# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub:
**Security > Report a vulnerability** on this repository
(`https://github.com/EquinoxWN/hybrid-search-service/security/advisories/new`). Do not open a public issue.

Include what you found, how to reproduce it, and the impact you expect. You will get an answer
within 7 days. Fixes are released as soon as they are ready, and you are credited unless you ask
not to be.

## Supported versions

This project is pre-1.0. Only the latest commit on `main` receives fixes.

## Scope

- **In scope:** SQL injection, the API leaking internal errors or listening beyond loopback,
  unbounded memory or time from crafted queries, corpus files or model files (the safetensors
  and vocabulary loaders), and scripts writing outside the repository's `.tmp/`.
- **Out of scope:** relevance quality, and the trust-authenticated local and CI databases, which
  are throwaway and bound to 127.0.0.1.

## How this repository protects itself

- Every SQL statement is parameterised; the embedding size and finite values are checked before
  writes; document fields have length limits and must be valid UTF-8.
- Queries are 1 to 512 bytes, k is 1 to 50, each request has a 10 s deadline; the server has read,
  write and idle timeouts, a 16 KiB header limit, refuses non-loopback listen addresses, and
  answers backend failures with a generic `500`.
- The model loader bounds the safetensors header (16 MiB) and file (1 GiB), checks every tensor's
  offsets against its shape, and refuses unexpected shapes or activations; the vocabulary file is
  limited to 4 MiB and inputs to 64 KiB of text and 256 tokens.
- Downloads (model files, PostgreSQL zip, pgvector source) are pinned and checked by SHA-256; the
  PostgreSQL Windows zip is not code-signed by its publisher, so the hash is the check. The CI
  database image is pinned by digest.
- Every GitHub Action is pinned to a full commit SHA, and workflows run with read-only
  permissions and without persisted credentials.
- Dependabot update pull requests are paused (`open-pull-requests-limit: 0`) so the repository keeps a single `main` branch; dependencies are updated by hand and checked by the CI audit.
- CI runs gofmt, go vet, tests with the race detector and govulncheck on every push and pull request.
