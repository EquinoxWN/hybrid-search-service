// Command hsearch builds the demo corpus, indexes it into PostgreSQL with pgvector, answers
// queries in keyword, vector or hybrid mode, serves an HTTP API and scores judged queries.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/EquinoxWN/hybrid-search-service/internal/corpus"
	"github.com/EquinoxWN/hybrid-search-service/internal/embed"
	"github.com/EquinoxWN/hybrid-search-service/internal/eval"
	"github.com/EquinoxWN/hybrid-search-service/internal/search"
	"github.com/EquinoxWN/hybrid-search-service/internal/store"
)

const usage = `usage: hsearch <command> [flags]

commands:
  corpus  build the demo corpus from the Go standard library's documentation
  index   embed the corpus and load it into PostgreSQL (pgvector)
  query   run one query in hybrid, keyword or vector mode
  serve   serve GET /search and GET /healthz on a loopback address
  eval    score every mode on judged queries (Recall@k, MRR@k, nDCG@k)

the database comes from -dsn or HSEARCH_DSN; the model directory from -model`

const maxDocs = 100_000

// flagOutput receives flag usage and parse errors.
var flagOutput io.Writer = os.Stderr

// main runs a command and maps errors to exit codes.
func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	code, err := run(ctx, os.Args[1:], os.Stdout)
	if err != nil {
		fmt.Fprintln(os.Stderr, "hsearch:", err)
	}
	os.Exit(code)
}

// run executes one command; exit 2 means a usage or runtime error.
func run(ctx context.Context, args []string, out io.Writer) (int, error) {
	if len(args) == 0 {
		return 2, errors.New(usage)
	}
	cmds := map[string]func(context.Context, []string, io.Writer) error{
		"corpus": cmdCorpus, "index": cmdIndex, "query": cmdQuery, "serve": cmdServe, "eval": cmdEval,
	}
	f, ok := cmds[args[0]]
	if !ok {
		return 2, fmt.Errorf("unknown command %q\n%s", args[0], usage)
	}
	if err := f(ctx, args[1:], out); err != nil {
		return 2, err
	}
	return 0, nil
}

type common struct {
	dsn, model *string
}

// newFlags creates a flag set with the shared database and model flags.
func newFlags(name string) (*flag.FlagSet, common) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(flagOutput)
	c := common{
		dsn:   fs.String("dsn", os.Getenv("HSEARCH_DSN"), "PostgreSQL connection string (default $HSEARCH_DSN)"),
		model: fs.String("model", ".tmp/models/all-MiniLM-L6-v2", "directory with config.json, vocab.txt, model.safetensors"),
	}
	return fs, c
}

// open loads the model and connects to the database.
func (c common) open(ctx context.Context) (*embed.Model, *store.Store, error) {
	if *c.dsn == "" {
		return nil, nil, errors.New("no database: set -dsn or HSEARCH_DSN")
	}
	m, err := embed.Load(*c.model)
	if err != nil {
		return nil, nil, err
	}
	st, err := store.Open(ctx, *c.dsn, m.Dim())
	if err != nil {
		return nil, nil, err
	}
	return m, st, nil
}

// cmdCorpus writes the standard-library corpus as JSON lines.
func cmdCorpus(_ context.Context, args []string, out io.Writer) error {
	fs := flag.NewFlagSet("corpus", flag.ContinueOnError)
	fs.SetOutput(flagOutput)
	goroot := fs.String("goroot", "", "Go installation whose src/ is read (go env GOROOT)")
	dst := fs.String("out", ".tmp/corpus.jsonl", "output file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *goroot == "" {
		return errors.New("corpus needs -goroot")
	}
	docs, err := corpus.Build(*goroot, corpus.Packages)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(*dst), 0o755); err != nil {
		return err
	}
	f, err := os.Create(filepath.Clean(*dst))
	if err != nil {
		return err
	}
	if err := errors.Join(corpus.WriteJSONL(f, docs), f.Close()); err != nil {
		return err
	}
	fmt.Fprintf(out, "corpus: %d documents from %d packages -> %s\n", len(docs), len(corpus.Packages), *dst)
	return nil
}

// readCorpus loads a corpus file.
func readCorpus(path string) ([]corpus.Doc, error) {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return corpus.ReadJSONL(f, maxDocs)
}

// cmdIndex embeds and stores the corpus.
func cmdIndex(ctx context.Context, args []string, out io.Writer) error {
	fs, c := newFlags("index")
	src := fs.String("corpus", ".tmp/corpus.jsonl", "corpus file from hsearch corpus")
	workers := fs.Int("workers", 8, "embedding goroutines (1..64)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *workers < 1 || *workers > 64 {
		return errors.New("workers must be 1..64")
	}
	docs, err := readCorpus(*src)
	if err != nil {
		return err
	}
	m, st, err := c.open(ctx)
	if err != nil {
		return err
	}
	defer st.Close()
	if err := st.Migrate(ctx); err != nil {
		return err
	}
	start := time.Now()
	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = d.Title + "\n" + d.Body
	}
	vecs := m.EmbedBatch(texts, *workers)
	embedTime := time.Since(start)
	start = time.Now()
	const batch = 500
	for i := 0; i < len(docs); i += batch {
		end := min(i+batch, len(docs))
		rows := make([]store.Doc, 0, end-i)
		for j := i; j < end; j++ {
			rows = append(rows, store.Doc{ID: docs[j].ID, Title: docs[j].Title, Body: docs[j].Body, Embedding: vecs[j]})
		}
		if err := st.Upsert(ctx, rows); err != nil {
			return err
		}
	}
	n, err := st.Count(ctx)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "index: embedded %d documents in %s (%.0f docs/s, %d workers), stored in %s; table holds %d\n",
		len(docs), embedTime.Round(time.Millisecond), float64(len(docs))/embedTime.Seconds(), *workers,
		time.Since(start).Round(time.Millisecond), n)
	return nil
}

// cmdQuery prints the results of one query.
func cmdQuery(ctx context.Context, args []string, out io.Writer) error {
	fs, c := newFlags("query")
	modeS := fs.String("mode", "hybrid", "hybrid, keyword or vector")
	k := fs.Int("k", 10, "results (1..50)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	mode, err := search.ParseMode(*modeS)
	if err != nil {
		return err
	}
	m, st, err := c.open(ctx)
	if err != nil {
		return err
	}
	defer st.Close()
	resp, err := search.New(m, st, 0).Search(ctx, strings.Join(fs.Args(), " "), mode, *k)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "%s %q in %.1f ms\n", resp.Mode, resp.Query, resp.TookMS)
	for i, r := range resp.Results {
		fmt.Fprintf(out, "%2d. %-40s score=%.4f keyword_rank=%d vector_rank=%d\n", i+1, r.ID, r.Score, r.KeywordRank, r.VectorRank)
	}
	return nil
}

// cmdServe runs the HTTP API until interrupted.
func cmdServe(ctx context.Context, args []string, out io.Writer) error {
	fs, c := newFlags("serve")
	addr := fs.String("addr", "127.0.0.1:8080", "listen address (loopback only)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	host, _, err := net.SplitHostPort(*addr)
	if ip := net.ParseIP(host); err != nil || (host != "localhost" && (ip == nil || !ip.IsLoopback())) {
		return fmt.Errorf("serve: %q is not a loopback address", *addr)
	}
	m, st, err := c.open(ctx)
	if err != nil {
		return err
	}
	defer st.Close()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	srv := &http.Server{
		Addr: *addr, Handler: search.Handler(search.New(m, st, 0), st, log),
		ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 15 * time.Second,
		IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 << 10,
	}
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "serving on http://%s\n", ln.Addr())
	errc := make(chan error, 1)
	go func() { errc <- srv.Serve(ln) }()
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
		shut, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(shut)
	}
}

// cmdEval scores the three modes on judged queries.
func cmdEval(ctx context.Context, args []string, out io.Writer) error {
	fs, c := newFlags("eval")
	queries := fs.String("queries", "testdata/queries.json", "judged queries")
	src := fs.String("corpus", ".tmp/corpus.jsonl", "corpus file, to check that every judged document exists")
	k := fs.Int("k", 10, "cutoff (1..50)")
	perQuery := fs.Bool("per-query", false, "also print the first relevant rank of each query per mode")
	if err := fs.Parse(args); err != nil {
		return err
	}
	qs, err := eval.Load(*queries)
	if err != nil {
		return err
	}
	docs, err := readCorpus(*src)
	if err != nil {
		return err
	}
	ids := map[string]bool{}
	for _, d := range docs {
		ids[d.ID] = true
	}
	if err := eval.CheckJudgments(qs, ids); err != nil {
		return err
	}
	m, st, err := c.open(ctx)
	if err != nil {
		return err
	}
	defer st.Close()
	scores, err := eval.Run(ctx, search.New(m, st, 0), qs, *k)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "%d judged queries, %d documents, k=%d\n", len(qs), len(docs), *k)
	fmt.Fprintf(out, "%-8s %9s %7s %8s %s\n", "mode", "Recall@k", "MRR@k", "nDCG@k", "queries with a hit")
	for _, s := range scores {
		fmt.Fprintf(out, "%-8s %9.3f %7.3f %8.3f %d/%d\n", s.Mode, s.Recall, s.MRR, s.NDCG, s.Hits, len(qs))
	}
	if *perQuery {
		fmt.Fprintf(out, "\nfirst relevant rank (0 = none in the top %d)\n%-4s %7s %6s %6s  %s\n", *k, "id", "keyword", "vector", "hybrid", "query")
		for i, q := range qs {
			fmt.Fprintf(out, "%-4s %7d %6d %6d  %s\n", q.ID, scores[0].First[i], scores[1].First[i], scores[2].First[i], q.Text)
		}
	}
	return nil
}
