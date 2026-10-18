package corpus

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
)

func TestBuildExtractsExportedDocumentedDeclarations(t *testing.T) {
	docs, err := Build("testdata/goroot", []string{"demo"})
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, d := range docs {
		ids = append(ids, d.ID)
	}
	want := []string{"demo", "demo.Add", "demo.NewWidget", "demo.Old", "demo.Widget", "demo.Widget.Rename"}
	if !reflect.DeepEqual(ids, want) {
		t.Fatalf("ids %v, want %v (no unexported, undocumented, Deprecated-only or Windows-only entries)", ids, want)
	}
	byID := map[string]Doc{}
	for _, d := range docs {
		byID[d.ID] = d
	}
	if got := byID["demo.Add"].Body; got != "func Add(a, b int) int\nAdd returns the sum of a and b." {
		t.Fatalf("Add body %q", got)
	}
	if got := byID["demo"].Body; strings.Contains(got, "second paragraph") || !strings.HasPrefix(got, "package demo\n") {
		t.Fatalf("package body %q", got)
	}
	if got := byID["demo.Widget"].Body; !strings.HasPrefix(got, "type Widget struct\n") {
		t.Fatalf("type body %q", got)
	}
	if got := byID["demo.Widget.Rename"].Body; !strings.HasPrefix(got, "func (w *Widget) Rename(name string)\n") {
		t.Fatalf("method body %q", got)
	}
}

func TestBuildRejectsBadPaths(t *testing.T) {
	for _, p := range []string{"../etc", "/abs", "missing"} {
		if _, err := Build("testdata/goroot", []string{p}); err == nil {
			t.Errorf("package %q accepted", p)
		}
	}
}

func TestJSONLRoundTripAndLimits(t *testing.T) {
	docs := []Doc{{ID: "a", Title: "a", Body: "x <b> & y"}, {ID: "b", Title: "b", Body: "é"}}
	var buf bytes.Buffer
	if err := WriteJSONL(&buf, docs); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "<b> &") {
		t.Fatal("HTML escaping should be off")
	}
	got, err := ReadJSONL(bytes.NewReader(buf.Bytes()), 10)
	if err != nil || !reflect.DeepEqual(got, docs) {
		t.Fatalf("round trip %v %v", got, err)
	}
	if _, err := ReadJSONL(bytes.NewReader(buf.Bytes()), 1); err == nil {
		t.Error("document limit not enforced")
	}
	for _, bad := range []string{`{"id":"a","extra":1}`, `{"title":"no id"}`, `not json`, `{"id":"` + strings.Repeat("x", 70<<10) + `"}`} {
		if _, err := ReadJSONL(strings.NewReader(bad+"\n"), 10); err == nil {
			t.Errorf("accepted %.30q", bad)
		}
	}
}

func TestTruncateKeepsRunes(t *testing.T) {
	if got := truncate("aé", 2); got != "a" {
		t.Fatalf("truncate cut a rune: %q", got)
	}
	if got := truncate("abc", 5); got != "abc" {
		t.Fatalf("truncate %q", got)
	}
}
