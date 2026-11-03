package fusion

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestRRFWorkedExample(t *testing.T) {
	got := RRF(60, 10,
		List{Name: "keyword", IDs: []string{"a", "b", "c"}},
		List{Name: "vector", IDs: []string{"c", "a", "d"}})
	ids := []string{}
	for _, f := range got {
		ids = append(ids, f.ID)
	}
	// a: 1/61+1/62, c: 1/63+1/61, b: 1/62, d: 1/63.
	if !slices.Equal(ids, []string{"a", "c", "b", "d"}) {
		t.Fatalf("order %v", ids)
	}
	if math.Abs(got[0].Score-(1.0/61+1.0/62)) > 1e-12 || !slices.Equal(got[0].Ranks, []int{1, 2}) || !slices.Equal(got[3].Ranks, []int{0, 3}) {
		t.Fatalf("scores or ranks wrong: %+v", got)
	}
}

func TestRRFEdgeCases(t *testing.T) {
	if got := RRF(60, 10); len(got) != 0 {
		t.Fatal("no lists should give no results")
	}
	got := RRF(0, 1, List{IDs: []string{"x", "x", "y"}})
	if len(got) != 1 || got[0].ID != "x" || math.Abs(got[0].Score-1.0/61) > 1e-12 {
		t.Fatalf("duplicates or default k: %+v", got)
	}
	// Equal scores: best single rank first, then ID.
	tie := RRF(60, 10, List{IDs: []string{"b"}}, List{IDs: []string{"a"}})
	if tie[0].ID != "a" || tie[1].ID != "b" {
		t.Fatalf("tie order %+v", tie)
	}
}

func TestRRFProperties(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for trial := range 500 {
		var lists []List
		for range 1 + r.IntN(3) {
			var ids []string
			for range r.IntN(20) {
				ids = append(ids, fmt.Sprint("d", r.IntN(30)))
			}
			lists = append(lists, List{IDs: ids})
		}
		limit := r.IntN(25)
		got := RRF(60, limit, lists...)
		if len(got) > limit {
			t.Fatalf("trial %d: %d results over limit %d", trial, len(got), limit)
		}
		seen := map[string]bool{}
		for i, f := range got {
			if seen[f.ID] {
				t.Fatalf("trial %d: duplicate %s", trial, f.ID)
			}
			seen[f.ID] = true
			if i > 0 && f.Score > got[i-1].Score {
				t.Fatalf("trial %d: not sorted by score", trial)
			}
			var want float64
			for li, rank := range f.Ranks {
				if rank > 0 {
					want += 1 / (60 + float64(rank))
					if lists[li].IDs[rank-1] != f.ID || slices.Index(lists[li].IDs, f.ID) != rank-1 {
						t.Fatalf("trial %d: rank %d of %s is not its first position", trial, rank, f.ID)
					}
				}
			}
			if math.Abs(want-f.Score) > 1e-12 {
				t.Fatalf("trial %d: score %v, want %v", trial, f.Score, want)
			}
		}
		// A document first in every list beats every other document.
		if len(lists) == 2 && len(lists[0].IDs) > 0 && len(lists[1].IDs) > 0 && lists[0].IDs[0] == lists[1].IDs[0] && limit > 0 {
			if got[0].ID != lists[0].IDs[0] {
				t.Fatalf("trial %d: top of both lists is not first", trial)
			}
		}
	}
}

func TestRRFEqualScoresPreferTheBetterSingleRank(t *testing.T) {
	// With k = 1, rank 1 in one list scores 1/2, exactly as much as ranks 2 and 5 in two others.
	got := RRF(1, 10, List{IDs: []string{"b"}}, List{IDs: []string{"x", "a"}}, List{IDs: []string{"y1", "y2", "y3", "y4", "a"}})
	pos := map[string]int{}
	for i, f := range got {
		pos[f.ID] = i
	}
	a, b := got[pos["a"]], got[pos["b"]]
	if a.Score != b.Score {
		t.Fatalf("test setup: scores %v and %v should be equal", a.Score, b.Score)
	}
	if pos["b"] > pos["a"] {
		t.Fatalf("b (best rank 1) should precede a (best rank 2) although a sorts first by ID: %+v", got)
	}
}
