// Package search answers queries in three modes: keyword (PostgreSQL full-text search), vector
// (pgvector) and hybrid, which runs both and merges them with reciprocal rank fusion.
package search

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/EquinoxWN/hybrid-search-service/internal/fusion"
	"github.com/EquinoxWN/hybrid-search-service/internal/store"
)

// Mode selects the retrieval method.
type Mode string

// Modes.
const (
	Hybrid  Mode = "hybrid"
	Keyword Mode = "keyword"
	Vector  Mode = "vector"
)

// Modes lists every mode in a fixed order.
var Modes = []Mode{Keyword, Vector, Hybrid}

// Query limits.
const (
	MaxQueryBytes = 512
	MaxK          = 50
	// Candidates is how many results each list contributes to the fusion.
	Candidates = 50
)

// ErrBadRequest marks errors caused by the query itself.
var ErrBadRequest = errors.New("bad request")

// Embedder turns text into a vector.
type Embedder interface {
	Embed(text string) []float32
}

// Backend runs the two ranked retrievals.
type Backend interface {
	Keyword(ctx context.Context, query string, k int) ([]store.Hit, error)
	Vector(ctx context.Context, emb []float32, k int) ([]store.Hit, error)
}

// Result is one ranked document; ranks are 1-based positions in each list (0 if absent).
type Result struct {
	ID          string  `json:"id"`
	Title       string  `json:"title"`
	Snippet     string  `json:"snippet"`
	Score       float64 `json:"score"`
	KeywordRank int     `json:"keyword_rank,omitempty"`
	VectorRank  int     `json:"vector_rank,omitempty"`
}

// Response is the answer to one query.
type Response struct {
	Query   string   `json:"query"`
	Mode    Mode     `json:"mode"`
	TookMS  float64  `json:"took_ms"`
	Results []Result `json:"results"`
}

// Service combines an embedder and a backend.
type Service struct {
	emb Embedder
	db  Backend
	rrf float64
}

// New returns a service using fusion constant rrfK (0 means fusion.DefaultK).
func New(emb Embedder, db Backend, rrfK float64) *Service {
	return &Service{emb: emb, db: db, rrf: rrfK}
}

// ParseMode validates a mode name; empty means hybrid.
func ParseMode(s string) (Mode, error) {
	switch Mode(s) {
	case "", Hybrid:
		return Hybrid, nil
	case Keyword, Vector:
		return Mode(s), nil
	}
	return "", fmt.Errorf("%w: mode must be hybrid, keyword or vector", ErrBadRequest)
}

// Search runs one query.
func (s *Service) Search(ctx context.Context, query string, mode Mode, k int) (Response, error) {
	start := time.Now()
	query = strings.TrimSpace(query)
	switch {
	case query == "" || len(query) > MaxQueryBytes || !utf8.ValidString(query):
		return Response{}, fmt.Errorf("%w: query must be 1..%d bytes of UTF-8", ErrBadRequest, MaxQueryBytes)
	case k < 1 || k > MaxK:
		return Response{}, fmt.Errorf("%w: k must be 1..%d", ErrBadRequest, MaxK)
	}
	var results []Result
	var err error
	switch mode {
	case Keyword:
		var hits []store.Hit
		if hits, err = s.db.Keyword(ctx, query, k); err == nil {
			results = fromHits(hits, true)
		}
	case Vector:
		var hits []store.Hit
		if hits, err = s.db.Vector(ctx, s.emb.Embed(query), k); err == nil {
			results = fromHits(hits, false)
		}
	case Hybrid:
		results, err = s.hybrid(ctx, query, k)
	default:
		return Response{}, fmt.Errorf("%w: unknown mode %q", ErrBadRequest, mode)
	}
	if err != nil {
		return Response{}, err
	}
	if results == nil {
		results = []Result{}
	}
	return Response{Query: query, Mode: mode, Results: results, TookMS: float64(time.Since(start).Microseconds()) / 1000}, nil
}

// hybrid runs the keyword search while the query is embedded, then fuses both lists.
func (s *Service) hybrid(ctx context.Context, query string, k int) ([]Result, error) {
	type kwResult struct {
		hits []store.Hit
		err  error
	}
	kwc := make(chan kwResult, 1)
	go func() {
		h, err := s.db.Keyword(ctx, query, Candidates)
		kwc <- kwResult{h, err}
	}()
	vec, verr := s.db.Vector(ctx, s.emb.Embed(query), Candidates)
	kw := <-kwc
	if err := errors.Join(kw.err, verr); err != nil {
		return nil, err
	}
	byID := map[string]store.Hit{}
	ids := func(hits []store.Hit) []string {
		out := make([]string, len(hits))
		for i, h := range hits {
			out[i] = h.ID
			byID[h.ID] = h
		}
		return out
	}
	fused := fusion.RRF(s.rrf, k, fusion.List{Name: "keyword", IDs: ids(kw.hits)}, fusion.List{Name: "vector", IDs: ids(vec)})
	out := make([]Result, len(fused))
	for i, f := range fused {
		h := byID[f.ID]
		out[i] = Result{ID: f.ID, Title: h.Title, Snippet: h.Snippet, Score: f.Score, KeywordRank: f.Ranks[0], VectorRank: f.Ranks[1]}
	}
	return out, nil
}

// fromHits converts one ranked list.
func fromHits(hits []store.Hit, keyword bool) []Result {
	out := make([]Result, len(hits))
	for i, h := range hits {
		out[i] = Result{ID: h.ID, Title: h.Title, Snippet: h.Snippet, Score: h.Score}
		if keyword {
			out[i].KeywordRank = i + 1
		} else {
			out[i].VectorRank = i + 1
		}
	}
	return out
}
