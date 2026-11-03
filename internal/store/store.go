// Package store keeps documents in PostgreSQL: a generated tsvector column with a GIN index for
// keyword search, and a pgvector column with an HNSW index (cosine distance) for vector search.
package store

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"
	"unicode/utf8"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Limits on stored and queried data.
const (
	MaxIDBytes    = 200
	MaxTitleBytes = 500
	MaxBodyBytes  = 8 << 10
	MaxK          = 200
	snippetChars  = 240
	efSearch      = 200 // HNSW candidate list; must be at least the largest k
	// connectTimeout bounds each new connection when the DSN sets no connect_timeout.
	connectTimeout = 10 * time.Second
)

// Doc is one document with its embedding.
type Doc struct {
	ID        string
	Title     string
	Body      string
	Embedding []float32
}

// Hit is one search result.
type Hit struct {
	ID      string
	Title   string
	Snippet string
	Score   float64
}

// Store is a connection pool bound to one embedding size.
type Store struct {
	pool *pgxpool.Pool
	dim  int
}

// Open connects and sets the HNSW search width on every connection.
func Open(ctx context.Context, dsn string, dim int) (*Store, error) {
	if dim <= 0 || dim > 16000 {
		return nil, fmt.Errorf("store: embedding size %d out of range", dim)
	}
	cfg, err := poolConfig(dsn)
	if err != nil {
		return nil, err
	}
	cfg.AfterConnect = func(ctx context.Context, c *pgx.Conn) error {
		_, err := c.Exec(ctx, "SET hnsw.ef_search = "+strconv.Itoa(efSearch))
		if err != nil && isUndefinedParam(err) {
			return nil // before CREATE EXTENSION the setting does not exist yet
		}
		return err
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("store: connect: %w", err)
	}
	return &Store{pool: pool, dim: dim}, nil
}

// poolConfig parses the DSN, caps the pool and bounds connection attempts.
func poolConfig(dsn string) (*pgxpool.Config, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("store: parse DSN: %w", err)
	}
	cfg.MaxConns = 8
	if cfg.ConnConfig.ConnectTimeout == 0 {
		cfg.ConnConfig.ConnectTimeout = connectTimeout
	}
	return cfg, nil
}

// isUndefinedParam reports PostgreSQL's "unrecognized configuration parameter" error.
func isUndefinedParam(err error) bool {
	var pgErr interface{ SQLState() string }
	return errors.As(err, &pgErr) && pgErr.SQLState() == "42704"
}

// Close releases the pool.
func (s *Store) Close() { s.pool.Close() }

// Ping checks the connection.
func (s *Store) Ping(ctx context.Context) error { return s.pool.Ping(ctx) }

// Migrate creates the extension, the table and both indexes if they do not exist.
func (s *Store) Migrate(ctx context.Context) error {
	stmts := []string{
		"CREATE EXTENSION IF NOT EXISTS vector",
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS documents (
			id text PRIMARY KEY,
			title text NOT NULL,
			body text NOT NULL,
			tsv tsvector GENERATED ALWAYS AS (
				setweight(to_tsvector('english', replace(replace(title, '.', ' '), '/', ' ')), 'A') ||
				setweight(to_tsvector('english', body), 'B')) STORED,
			embedding vector(%d) NOT NULL)`, s.dim),
		"CREATE INDEX IF NOT EXISTS documents_tsv ON documents USING gin (tsv)",
		"CREATE INDEX IF NOT EXISTS documents_embedding ON documents USING hnsw (embedding vector_cosine_ops)",
	}
	for _, q := range stmts {
		if _, err := s.pool.Exec(ctx, q); err != nil {
			return fmt.Errorf("store: migrate: %w", err)
		}
	}
	// New connections pick up hnsw.ef_search now that the extension exists.
	s.pool.Reset()
	return nil
}

// validate checks one document before it is written.
func (s *Store) validate(d Doc) error {
	switch {
	case d.ID == "" || len(d.ID) > MaxIDBytes || len(d.Title) > MaxTitleBytes || len(d.Body) > MaxBodyBytes:
		return fmt.Errorf("store: document %.40q: empty ID or field too long", d.ID)
	case !utf8.ValidString(d.ID) || !utf8.ValidString(d.Title) || !utf8.ValidString(d.Body):
		return fmt.Errorf("store: document %.40q: invalid UTF-8", d.ID)
	}
	return s.checkVector(d.Embedding)
}

// checkVector requires the configured size and finite values.
func (s *Store) checkVector(v []float32) error {
	if len(v) != s.dim {
		return fmt.Errorf("store: embedding has %d values, want %d", len(v), s.dim)
	}
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return errors.New("store: embedding has a non-finite value")
		}
	}
	return nil
}

// Upsert inserts or replaces documents in one batch.
func (s *Store) Upsert(ctx context.Context, docs []Doc) error {
	b := &pgx.Batch{}
	for _, d := range docs {
		if err := s.validate(d); err != nil {
			return err
		}
		b.Queue(`INSERT INTO documents (id, title, body, embedding) VALUES ($1, $2, $3, $4::vector)
			ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, body = EXCLUDED.body, embedding = EXCLUDED.embedding`,
			d.ID, d.Title, d.Body, vectorLiteral(d.Embedding))
	}
	return s.pool.SendBatch(ctx, b).Close()
}

// Count returns the number of documents.
func (s *Store) Count(ctx context.Context) (int, error) {
	var n int
	err := s.pool.QueryRow(ctx, "SELECT count(*) FROM documents").Scan(&n)
	return n, err
}

// Keyword ranks documents matching every term of a web-search style query by ts_rank_cd.
// Requiring all terms keeps the keyword list short and precise; OR-ing them was measured and
// lowered hybrid nDCG@10 from 0.471 to 0.405 (ADR 0003).
func (s *Store) Keyword(ctx context.Context, query string, k int) ([]Hit, error) {
	if k < 1 || k > MaxK {
		return nil, fmt.Errorf("store: k must be 1..%d", MaxK)
	}
	return s.hits(ctx, `SELECT id, title, left(body, $3), ts_rank_cd(tsv, q) AS score
		FROM documents, websearch_to_tsquery('english', $1) q
		WHERE tsv @@ q ORDER BY score DESC, id LIMIT $2`, query, k, snippetChars)
}

// Vector ranks documents by cosine similarity to emb through the HNSW index.
func (s *Store) Vector(ctx context.Context, emb []float32, k int) ([]Hit, error) {
	if k < 1 || k > MaxK {
		return nil, fmt.Errorf("store: k must be 1..%d", MaxK)
	}
	if err := s.checkVector(emb); err != nil {
		return nil, err
	}
	return s.hits(ctx, `SELECT id, title, left(body, $3), 1 - (embedding <=> $1::vector) AS score
		FROM documents ORDER BY embedding <=> $1::vector LIMIT $2`, vectorLiteral(emb), k, snippetChars)
}

// hits runs a ranking query.
func (s *Store) hits(ctx context.Context, sql string, args ...any) ([]Hit, error) {
	rows, err := s.pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(r pgx.CollectableRow) (Hit, error) {
		var h Hit
		err := r.Scan(&h.ID, &h.Title, &h.Snippet, &h.Score)
		return h, err
	})
}

// vectorLiteral formats a vector in pgvector's text form.
func vectorLiteral(v []float32) string {
	b := make([]byte, 0, len(v)*12)
	b = append(b, '[')
	for i, x := range v {
		if i > 0 {
			b = append(b, ',')
		}
		b = strconv.AppendFloat(b, float64(x), 'g', -1, 32)
	}
	return string(append(b, ']'))
}
