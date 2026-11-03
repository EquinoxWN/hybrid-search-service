package eval

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/EquinoxWN/hybrid-search-service/internal/search"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestMetrics(t *testing.T) {
	r, rr, nd := Metrics([]string{"x", "a", "y", "b"}, []string{"a", "b"}, 10)
	idcg := 1 + 1/math.Log2(3)
	if !near(r, 1) || !near(rr, 0.5) || !near(nd, (1/math.Log2(3)+1/math.Log2(5))/idcg) {
		t.Fatalf("got %v %v %v", r, rr, nd)
	}
	r, rr, nd = Metrics([]string{"x", "a", "a"}, []string{"a", "b"}, 1)
	if r != 0 || rr != 0 || nd != 0 {
		t.Fatalf("cutoff 1: %v %v %v", r, rr, nd)
	}
	if r, rr, nd = Metrics([]string{"a", "a"}, []string{"a"}, 10); !near(r, 1) || !near(rr, 1) || !near(nd, 1) {
		t.Fatalf("duplicates must count once: %v %v %v", r, rr, nd)
	}
	if r, _, _ := Metrics(nil, nil, 10); r != 0 {
		t.Fatal("no judgments")
	}
}

type fixed map[search.Mode][]string

func (f fixed) Search(_ context.Context, _ string, m search.Mode, _ int) (search.Response, error) {
	var res []search.Result
	for _, id := range f[m] {
		res = append(res, search.Result{ID: id})
	}
	return search.Response{Results: res}, nil
}

func TestRunAveragesEveryMode(t *testing.T) {
	qs := []Query{{ID: "1", Text: "q", Relevant: []string{"a"}}, {ID: "2", Text: "q", Relevant: []string{"z"}}}
	scores, err := Run(context.Background(), fixed{search.Keyword: {"a"}, search.Vector: {"b", "a"}, search.Hybrid: nil}, qs, 10)
	if err != nil || len(scores) != 3 {
		t.Fatal(err)
	}
	if s := scores[0]; s.Mode != search.Keyword || !near(s.MRR, 0.5) || !near(s.Recall, 0.5) || s.Hits != 1 {
		t.Fatalf("keyword %+v", s)
	}
	if s := scores[1]; !near(s.MRR, 0.25) || s.Hits != 1 {
		t.Fatalf("vector %+v", s)
	}
	if s := scores[2]; s.MRR != 0 || s.Hits != 0 {
		t.Fatalf("hybrid %+v", s)
	}
}

func TestLoadAndCheckJudgments(t *testing.T) {
	qs, err := Load(filepath.Join("..", "..", "testdata", "queries.json"))
	if err != nil || len(qs) < 30 {
		t.Fatalf("queries.json: %d queries, %v", len(qs), err)
	}
	dir := t.TempDir()
	for name, body := range map[string]string{
		"dup.json":     `[{"id":"a","text":"t","relevant":["x"]},{"id":"a","text":"t","relevant":["x"]}]`,
		"empty.json":   `[{"id":"a","text":"t","relevant":[]}]`,
		"unknown.json": `[{"id":"a","text":"t","relevant":["x"],"extra":1}]`,
		"none.json":    `[]`,
	} {
		p := filepath.Join(dir, name)
		_ = os.WriteFile(p, []byte(body), 0o600)
		if _, err := Load(p); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	err = CheckJudgments([]Query{{ID: "q", Relevant: []string{"a", "typo"}}}, map[string]bool{"a": true})
	if err == nil || !strings.Contains(err.Error(), `"typo" is not in the corpus`) {
		t.Fatalf("missing judgment: %v", err)
	}
}
