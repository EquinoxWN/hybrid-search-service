// Package fusion merges ranked result lists with reciprocal rank fusion (Cormack, Clarke and
// Buettcher, SIGIR 2009): each document scores the sum of 1/(k + rank) over the lists it appears in.
package fusion

import "sort"

// DefaultK is the constant from the original paper; it damps the weight of the top ranks.
const DefaultK = 60

// List is one ranked list of document IDs, best first.
type List struct {
	Name string
	IDs  []string
}

// Fused is a document with its fused score and its 1-based rank in each input list (0 if absent).
type Fused struct {
	ID    string
	Score float64
	Ranks []int
}

// RRF fuses the lists and returns the best limit documents. Duplicates within a list count once,
// at their best rank. Ties are broken by the best single rank, then by ID, so the order is stable.
func RRF(k float64, limit int, lists ...List) []Fused {
	if k <= 0 {
		k = DefaultK
	}
	byID := map[string]*Fused{}
	var order []*Fused
	for li, l := range lists {
		for pos, id := range l.IDs {
			f := byID[id]
			if f == nil {
				f = &Fused{ID: id, Ranks: make([]int, len(lists))}
				byID[id] = f
				order = append(order, f)
			}
			if f.Ranks[li] != 0 {
				continue
			}
			f.Ranks[li] = pos + 1
			f.Score += 1 / (k + float64(pos+1))
		}
	}
	sort.Slice(order, func(i, j int) bool {
		a, b := order[i], order[j]
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		if ba, bb := bestRank(a.Ranks), bestRank(b.Ranks); ba != bb {
			return ba < bb
		}
		return a.ID < b.ID
	})
	if limit >= 0 && len(order) > limit {
		order = order[:limit]
	}
	out := make([]Fused, len(order))
	for i, f := range order {
		out[i] = *f
	}
	return out
}

// bestRank is the smallest non-zero rank.
func bestRank(ranks []int) int {
	best := int(^uint(0) >> 1)
	for _, r := range ranks {
		if r > 0 && r < best {
			best = r
		}
	}
	return best
}
