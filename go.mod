module github.com/EquinoxWN/hybrid-search-service

go 1.26.0

// Patched standard library (GO-2026-6809 is fixed in 1.26.10); older 1.26 toolchains switch to it.
toolchain go1.26.10

require (
	github.com/jackc/pgx/v5 v5.11.0
	golang.org/x/text v0.43.0
)

require (
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.3 // indirect
	golang.org/x/sync v0.24.0 // indirect
)
