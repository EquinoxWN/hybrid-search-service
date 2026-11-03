// Package eval scores retrieval modes on queries with known relevant documents: Recall@k,
// MRR@k and nDCG@k with binary relevance.
package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/EquinoxWN/hybrid-search-service/internal/search"
)

// Query is one judged query.
type Query struct {
	ID       string   `json:"id"`
	Text     string   `json:"text"`
	Relevant []string `json:"relevant"`
}

// Scores are averages over all queries.
type Scores struct {
	Mode   search.Mode
	Recall float64
	MRR    float64
	NDCG   float64
	Hits   int   // queries with at least one relevant result in the top k
	First  []int // per query: rank of the first relevant result, 0 if none in the top k
}

// Searcher is the part of the service evaluation needs.
type Searcher interface {
	Search(ctx context.Context, q string, mode search.Mode, k int) (search.Response, error)
}

// Load reads and validates a judged query file.
func Load(path string) ([]Query, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var qs []Query
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&qs); err != nil {
		return nil, fmt.Errorf("eval: %s: %w", path, err)
	}
	if len(qs) == 0 {
		return nil, fmt.Errorf("eval: %s: no judged queries", path) // averages over zero queries are NaN
	}
	seen := map[string]bool{}
	for _, q := range qs {
		if q.ID == "" || q.Text == "" || len(q.Relevant) == 0 || seen[q.ID] {
			return nil, fmt.Errorf("eval: query %q: needs a unique id, text and relevant documents", q.ID)
		}
		seen[q.ID] = true
	}
	return qs, nil
}

// CheckJudgments fails if a relevant document is missing from the corpus, so a typo cannot
// silently lower every score.
func CheckJudgments(qs []Query, ids map[string]bool) error {
	var errs []error
	for _, q := range qs {
		for _, r := range q.Relevant {
			if !ids[r] {
				errs = append(errs, fmt.Errorf("eval: query %s: relevant document %q is not in the corpus", q.ID, r))
			}
		}
	}
	return errors.Join(errs...)
}

// Run scores every mode at cutoff k.
func Run(ctx context.Context, s Searcher, qs []Query, k int) ([]Scores, error) {
	var out []Scores
	for _, m := range search.Modes {
		sc := Scores{Mode: m}
		for _, q := range qs {
			resp, err := s.Search(ctx, q.Text, m, k)
			if err != nil {
				return nil, fmt.Errorf("eval: %s %s: %w", m, q.ID, err)
			}
			ids := make([]string, len(resp.Results))
			for i, r := range resp.Results {
				ids[i] = r.ID
			}
			r, rr, nd := Metrics(ids, q.Relevant, k)
			sc.Recall += r
			sc.MRR += rr
			sc.NDCG += nd
			first := 0
			if rr > 0 {
				sc.Hits++
				first = int(1/rr + 0.5)
			}
			sc.First = append(sc.First, first)
		}
		n := float64(len(qs))
		sc.Recall, sc.MRR, sc.NDCG = sc.Recall/n, sc.MRR/n, sc.NDCG/n
		out = append(out, sc)
	}
	return out, nil
}

// Metrics returns Recall@k, reciprocal rank of the first relevant result within k, and nDCG@k.
func Metrics(ranked, relevant []string, k int) (recall, rr, ndcg float64) {
	rel := map[string]bool{}
	for _, r := range relevant {
		rel[r] = true
	}
	found := map[string]bool{}
	var dcg float64
	for i, id := range ranked {
		if i >= k {
			break
		}
		if rel[id] && !found[id] {
			found[id] = true
			dcg += 1 / math.Log2(float64(i+2))
			if rr == 0 {
				rr = 1 / float64(i+1)
			}
		}
	}
	var ideal float64
	for i := range min(len(rel), k) {
		ideal += 1 / math.Log2(float64(i+2))
	}
	if len(rel) == 0 {
		return 0, 0, 0
	}
	return float64(len(found)) / float64(len(rel)), rr, dcg / ideal
}
