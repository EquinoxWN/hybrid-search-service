// Package corpus builds the demo document set from the Go standard library's own documentation:
// one document per package, exported function, type and method, with its signature and the first
// paragraph of its doc comment. The text is BSD-licensed and is regenerated from the pinned Go
// toolchain, so the corpus needs no download. Files are selected for linux/amd64 on every
// platform, so the corpus is the same on Windows and in CI.
package corpus

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/parser"
	"go/printer"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// Doc is one corpus entry.
type Doc struct {
	ID    string `json:"id"`
	Title string `json:"title"`
	Body  string `json:"body"`
}

// Packages is the default set: the standard-library packages most programs use.
var Packages = []string{
	"archive/zip", "bufio", "bytes", "compress/gzip", "container/heap", "container/list", "context",
	"crypto/rand", "crypto/sha256", "database/sql", "encoding/base64", "encoding/csv", "encoding/hex",
	"encoding/json", "errors", "flag", "fmt", "hash/crc32", "html/template", "io", "io/fs", "log",
	"log/slog", "maps", "math", "math/rand/v2", "mime/multipart", "net", "net/http", "net/url", "os",
	"os/exec", "os/signal", "path/filepath", "regexp", "slices", "sort", "strconv", "strings", "sync",
	"sync/atomic", "testing", "text/template", "time", "unicode", "unicode/utf8",
}

const maxBody = 700

// Build parses each package under goroot/src and returns its documents sorted by ID.
func Build(goroot string, pkgs []string) ([]Doc, error) {
	ctx := build.Default
	ctx.GOROOT, ctx.GOOS, ctx.GOARCH, ctx.CgoEnabled = goroot, "linux", "amd64", false
	seen := map[string]bool{}
	var out []Doc
	for _, path := range pkgs {
		docs, err := buildPackage(&ctx, goroot, path)
		if err != nil {
			return nil, err
		}
		for _, d := range docs {
			if !seen[d.ID] {
				seen[d.ID] = true
				out = append(out, d)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// buildPackage extracts the exported, documented declarations of one package.
func buildPackage(ctx *build.Context, goroot, path string) ([]Doc, error) {
	if strings.Contains(path, "..") || strings.HasPrefix(path, "/") {
		return nil, fmt.Errorf("corpus: bad package path %q", path)
	}
	dir := filepath.Join(goroot, "src", filepath.FromSlash(path))
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("corpus: %s: %w", path, err)
	}
	fset := token.NewFileSet()
	var files []*ast.File
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		if ok, err := ctx.MatchFile(dir, name); err != nil || !ok {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			return nil, fmt.Errorf("corpus: %s: %w", name, err)
		}
		files = append(files, f)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("corpus: no Go files for %s", path)
	}
	p, err := doc.NewFromFiles(fset, files, path)
	if err != nil {
		return nil, err
	}
	var out []Doc
	add := func(id, sig, text string) {
		if para := firstParagraph(text); para != "" && !strings.HasPrefix(para, "Deprecated:") {
			body := para
			if sig != "" {
				body = sig + "\n" + para
			}
			out = append(out, Doc{ID: id, Title: id, Body: truncate(body, maxBody)})
		}
	}
	add(path, "package "+p.Name, p.Doc)
	for _, f := range p.Funcs {
		add(path+"."+f.Name, signature(fset, f.Decl), f.Doc)
	}
	for _, t := range p.Types {
		add(path+"."+t.Name, typeLine(t), t.Doc)
		for _, f := range t.Funcs {
			add(path+"."+f.Name, signature(fset, f.Decl), f.Doc)
		}
		for _, m := range t.Methods {
			add(path+"."+t.Name+"."+m.Name, signature(fset, m.Decl), m.Doc)
		}
	}
	return out, nil
}

// signature prints a function declaration without its body.
func signature(fset *token.FileSet, d *ast.FuncDecl) string {
	cp := *d
	cp.Body, cp.Doc = nil, nil
	var b bytes.Buffer
	if err := printer.Fprint(&b, fset, &cp); err != nil {
		return ""
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

// typeLine names the kind of a type declaration.
func typeLine(t *doc.Type) string {
	for _, s := range t.Decl.Specs {
		if ts, ok := s.(*ast.TypeSpec); ok {
			switch ts.Type.(type) {
			case *ast.StructType:
				return "type " + t.Name + " struct"
			case *ast.InterfaceType:
				return "type " + t.Name + " interface"
			}
		}
	}
	return "type " + t.Name
}

// firstParagraph returns the doc comment up to the first blank line, on one line.
func firstParagraph(text string) string {
	para, _, _ := strings.Cut(strings.TrimSpace(text), "\n\n")
	return strings.Join(strings.Fields(para), " ")
}

// truncate cuts s to at most n bytes on a rune boundary.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// WriteJSONL writes one document per line.
func WriteJSONL(w io.Writer, docs []Doc) error {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	for _, d := range docs {
		if err := enc.Encode(d); err != nil {
			return err
		}
	}
	return nil
}

// ReadJSONL reads documents written by WriteJSONL, refusing unknown fields and oversized lines.
func ReadJSONL(r io.Reader, maxDocs int) ([]Doc, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), 64<<10)
	var out []Doc
	for line := 1; sc.Scan(); line++ {
		if len(out) >= maxDocs {
			return nil, fmt.Errorf("corpus: more than %d documents", maxDocs)
		}
		var d Doc
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&d); err != nil {
			return nil, fmt.Errorf("corpus: line %d: %w", line, err)
		}
		if d.ID == "" {
			return nil, fmt.Errorf("corpus: line %d: empty id", line)
		}
		out = append(out, d)
	}
	if err := sc.Err(); err != nil {
		return nil, errors.Join(errors.New("corpus: read"), err)
	}
	return out, nil
}
