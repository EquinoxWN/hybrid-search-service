package store

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestVectorLiteral(t *testing.T) {
	if got := vectorLiteral([]float32{1, -0.5, 0.1, 3e-8}); got != "[1,-0.5,0.1,3e-08]" {
		t.Fatalf("literal %s", got)
	}
}

func TestValidateRejectsBadDocuments(t *testing.T) {
	s := &Store{dim: 3}
	ok := Doc{ID: "a", Title: "t", Body: "b", Embedding: []float32{1, 0, 0}}
	if err := s.validate(ok); err != nil {
		t.Fatal(err)
	}
	bad := []Doc{
		{ID: "", Embedding: ok.Embedding},
		{ID: strings.Repeat("x", MaxIDBytes+1), Embedding: ok.Embedding},
		{ID: "a", Body: strings.Repeat("x", MaxBodyBytes+1), Embedding: ok.Embedding},
		{ID: "a\xff", Embedding: ok.Embedding},
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "a", Embedding: []float32{1, float32(math.NaN()), 0}},
		{ID: "a", Embedding: []float32{1, float32(math.Inf(1)), 0}},
	}
	for i, d := range bad {
		if err := s.validate(d); err == nil {
			t.Errorf("document %d accepted", i)
		}
	}
	if _, err := Open(context.Background(), "postgres://x", 0); err == nil {
		t.Error("dim 0 accepted")
	}
}

// openTest connects to the dedicated test database and recreates the table with 3 dimensions.
func openTest(t *testing.T) *Store {
	t.Helper()
	dsn := os.Getenv("HSEARCH_TEST_DSN")
	if dsn == "" {
		t.Skip("HSEARCH_TEST_DSN not set (make test-db, or CI's pgvector service)")
	}
	ctx := context.Background()
	s, err := Open(ctx, dsn, 3)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	if _, err := s.pool.Exec(ctx, "DROP TABLE IF EXISTS documents"); err != nil {
		t.Fatal(err)
	}
	if err := s.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	return s
}

func TestPostgresKeywordAndVectorSearch(t *testing.T) {
	s := openTest(t)
	ctx := context.Background()
	docs := []Doc{
		{ID: "strconv.Atoi", Title: "strconv.Atoi", Body: "Atoi is equivalent to ParseInt(s, 10, 0), converted to type int.", Embedding: []float32{1, 0, 0}},
		{ID: "time.Sleep", Title: "time.Sleep", Body: "Sleep pauses the current goroutine for at least the duration d.", Embedding: []float32{0, 1, 0}},
		{ID: "strings.Join", Title: "strings.Join", Body: "Join concatenates the elements of its first argument to create a single string.", Embedding: []float32{0.6, 0.8, 0}},
	}
	if err := s.Upsert(ctx, docs); err != nil {
		t.Fatal(err)
	}
	docs[1].Body = "Sleep pauses the current goroutine for at least the duration d. Updated."
	if err := s.Upsert(ctx, docs[1:2]); err != nil {
		t.Fatal(err)
	}
	if n, err := s.Count(ctx); err != nil || n != 3 {
		t.Fatalf("count %d %v", n, err)
	}
	kw, err := s.Keyword(ctx, "pauses goroutines", 10)
	if err != nil || len(kw) != 1 || kw[0].ID != "time.Sleep" || !strings.HasSuffix(kw[0].Snippet, "Updated.") {
		t.Fatalf("keyword %+v %v", kw, err)
	}
	// The title is indexed with dots split, so the identifier is searchable.
	if kw, _ := s.Keyword(ctx, "Atoi", 10); len(kw) != 1 || kw[0].ID != "strconv.Atoi" {
		t.Fatalf("title search %+v", kw)
	}
	if kw, _ := s.Keyword(ctx, "'; DROP TABLE documents; --", 10); len(kw) != 0 {
		t.Fatalf("hostile query %+v", kw)
	}
	vec, err := s.Vector(ctx, []float32{0, 1, 0}, 2)
	if err != nil || len(vec) != 2 || vec[0].ID != "time.Sleep" || vec[1].ID != "strings.Join" || math.Abs(vec[0].Score-1) > 1e-6 {
		t.Fatalf("vector %+v %v", vec, err)
	}
	if _, err := s.Vector(ctx, []float32{1, 0}, 2); err == nil {
		t.Fatal("wrong-size query vector accepted")
	}
	if _, err := s.Keyword(ctx, "x", 0); err == nil {
		t.Fatal("k=0 accepted")
	}
	if err := s.Ping(ctx); err != nil {
		t.Fatal(err)
	}
}

func TestPoolConfigBoundsConnects(t *testing.T) {
	cfg, err := poolConfig("postgres://u@127.0.0.1:1/db")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ConnConfig.ConnectTimeout != connectTimeout || cfg.MaxConns != 8 {
		t.Fatalf("connect timeout %s, max conns %d", cfg.ConnConfig.ConnectTimeout, cfg.MaxConns)
	}
	cfg, err = poolConfig("postgres://u@127.0.0.1:1/db?connect_timeout=3")
	if err != nil || cfg.ConnConfig.ConnectTimeout != 3*time.Second {
		t.Fatalf("the DSN's connect_timeout must win: %v %v", cfg.ConnConfig.ConnectTimeout, err)
	}
	if _, err := poolConfig("postgres://u@127.0.0.1:1/db?connect_timeout=x"); err == nil {
		t.Fatal("bad DSN accepted")
	}
}
