package search

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/EquinoxWN/hybrid-search-service/internal/store"
)

type fakeEmbedder struct{ calls atomic.Int32 }

func (f *fakeEmbedder) Embed(string) []float32 { f.calls.Add(1); return []float32{1, 0} }

// fakeBackend returns fixed lists and records the k it was asked for.
type fakeBackend struct {
	kw, vec []store.Hit
	err     error
	kwK     int
	pingErr error
}

func (f *fakeBackend) Keyword(_ context.Context, _ string, k int) ([]store.Hit, error) {
	f.kwK = k
	return f.kw, f.err
}

func (f *fakeBackend) Vector(context.Context, []float32, int) ([]store.Hit, error) {
	return f.vec, f.err
}

func (f *fakeBackend) Ping(context.Context) error { return f.pingErr }

func hits(ids ...string) []store.Hit {
	out := make([]store.Hit, len(ids))
	for i, id := range ids {
		out[i] = store.Hit{ID: id, Title: "t-" + id, Snippet: "s-" + id, Score: 1}
	}
	return out
}

func TestHybridFusesBothLists(t *testing.T) {
	be := &fakeBackend{kw: hits("a", "b", "c"), vec: hits("c", "a", "d")}
	emb := &fakeEmbedder{}
	resp, err := New(emb, be, 0).Search(context.Background(), "  query  ", Hybrid, 3)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, r := range resp.Results {
		ids = append(ids, r.ID)
	}
	if strings.Join(ids, ",") != "a,c,b" || resp.Query != "query" || be.kwK != Candidates || emb.calls.Load() != 1 {
		t.Fatalf("ids %v query %q kwK %d", ids, resp.Query, be.kwK)
	}
	if r := resp.Results[1]; r.KeywordRank != 3 || r.VectorRank != 1 || r.Title != "t-c" || r.Snippet != "s-c" {
		t.Fatalf("result %+v", r)
	}
}

func TestSingleModesAndEmptyResults(t *testing.T) {
	be := &fakeBackend{kw: hits("k1"), vec: hits("v1", "v2")}
	s := New(&fakeEmbedder{}, be, 0)
	kw, _ := s.Search(context.Background(), "q", Keyword, 5)
	vec, _ := s.Search(context.Background(), "q", Vector, 5)
	if kw.Results[0].KeywordRank != 1 || vec.Results[1].VectorRank != 2 || len(vec.Results) != 2 {
		t.Fatalf("keyword %+v vector %+v", kw, vec)
	}
	empty, _ := New(&fakeEmbedder{}, &fakeBackend{}, 0).Search(context.Background(), "q", Hybrid, 5)
	if empty.Results == nil {
		t.Fatal("empty results must encode as [] not null")
	}
}

func TestQueryValidation(t *testing.T) {
	s := New(&fakeEmbedder{}, &fakeBackend{}, 0)
	for _, c := range []struct {
		q    string
		mode Mode
		k    int
	}{{"", Hybrid, 10}, {"   ", Hybrid, 10}, {strings.Repeat("x", MaxQueryBytes+1), Hybrid, 10}, {"\xff", Hybrid, 10}, {"q", Hybrid, 0}, {"q", Hybrid, MaxK + 1}, {"q", "other", 10}} {
		if _, err := s.Search(context.Background(), c.q, c.mode, c.k); !errors.Is(err, ErrBadRequest) {
			t.Errorf("%.20q %s %d: %v", c.q, c.mode, c.k, err)
		}
	}
	if _, err := ParseMode("bogus"); !errors.Is(err, ErrBadRequest) {
		t.Error("bad mode accepted")
	}
	if m, _ := ParseMode(""); m != Hybrid {
		t.Error("empty mode should mean hybrid")
	}
	failing := New(&fakeEmbedder{}, &fakeBackend{err: errors.New("db down")}, 0)
	if _, err := failing.Search(context.Background(), "q", Hybrid, 5); err == nil || errors.Is(err, ErrBadRequest) {
		t.Fatalf("backend error: %v", err)
	}
}

// get calls the handler and decodes the JSON body.
func get(t *testing.T, h http.Handler, path string) (int, map[string]any) {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("%s: body %q is not JSON", path, rec.Body.String())
	}
	if rec.Header().Get("Content-Type") != "application/json" || rec.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Fatalf("%s: headers %v", path, rec.Header())
	}
	return rec.Code, body
}

func TestHTTPHandler(t *testing.T) {
	be := &fakeBackend{kw: hits("a"), vec: hits("a", "b")}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	h := Handler(New(&fakeEmbedder{}, be, 0), be, log)
	code, body := get(t, h, "/search?q="+url.QueryEscape("read a file")+"&k=2")
	if code != 200 || body["mode"] != "hybrid" || len(body["results"].([]any)) != 2 {
		t.Fatalf("search: %d %v", code, body)
	}
	for _, path := range []string{"/search?q=x&k=abc", "/search?q=x&mode=fuzzy", "/search?q=", "/search?q=x&k=999"} {
		if code, body := get(t, h, path); code != 400 || body["error"] == "" {
			t.Errorf("%s: %d %v", path, code, body)
		}
	}
	if code, _ := get(t, h, "/healthz"); code != 200 {
		t.Fatalf("healthz %d", code)
	}
	be.err, be.pingErr = errors.New("connection refused to 10.0.0.5"), errors.New("down")
	code, body = get(t, h, "/search?q=x")
	if code != 500 || body["error"] != "search failed" {
		t.Fatalf("backend failure must not leak details: %d %v", code, body)
	}
	if code, _ := get(t, h, "/healthz"); code != 503 {
		t.Fatalf("healthz when down: %d", code)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/search?q=x", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /search: %d", rec.Code)
	}
}

func TestQueryAtTheByteLimitIsAccepted(t *testing.T) {
	s := New(&fakeEmbedder{}, &fakeBackend{kw: hits("a")}, 0)
	if _, err := s.Search(context.Background(), strings.Repeat("x", MaxQueryBytes), Keyword, MaxK); err != nil {
		t.Fatalf("query of exactly %d bytes and k = %d refused: %v", MaxQueryBytes, MaxK, err)
	}
}
