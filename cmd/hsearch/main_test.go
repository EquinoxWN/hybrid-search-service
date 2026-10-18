package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCorpusCommand(t *testing.T) {
	// The fixture GOROOT holds one package; Build needs every listed package, so check the error
	// path with it and the success path through the corpus package's own tests.
	var out bytes.Buffer
	code, err := run(context.Background(), []string{"corpus", "-goroot", "../../internal/corpus/testdata/goroot", "-out", filepath.Join(t.TempDir(), "c.jsonl")}, &out)
	if code != 2 || err == nil || !strings.Contains(err.Error(), "archive/zip") {
		t.Fatalf("corpus with an incomplete GOROOT: %d %v", code, err)
	}
}

func TestUsageErrors(t *testing.T) {
	flagOutput = io.Discard
	t.Cleanup(func() { flagOutput = os.Stderr })
	t.Setenv("HSEARCH_DSN", "")
	for _, args := range [][]string{
		nil,
		{"nope"},
		{"corpus"},
		{"index", "-workers", "0"},
		{"index", "-corpus", "missing.jsonl"},
		{"query", "-mode", "fuzzy", "x"},
		{"query", "x"},
		{"serve", "-addr", "0.0.0.0:8080"},
		{"serve", "-addr", "example.com:80"},
		{"eval", "-queries", "missing.json"},
	} {
		var out bytes.Buffer
		if code, err := run(context.Background(), args, &out); code != 2 || err == nil {
			t.Errorf("%v: code %d err %v", args, code, err)
		}
	}
}
