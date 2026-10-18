package embed

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestBasicTokenize(t *testing.T) {
	cases := map[string][]string{
		"Hello, World!":       {"hello", ",", "world", "!"},
		"Café crème brûlée":   {"cafe", "creme", "brulee"},
		"a\x00b\tc\u00a0d":    {"ab", "c", "d"},
		"数据库":                 {"数", "据", "库"},
		"don't stop":          {"don", "'", "t", "stop"},
		"x.Y/z?id=4":          {"x", ".", "y", "/", "z", "?", "id", "=", "4"},
		"  \n ":               nil,
		"Ångström\u200bunits": {"angstromunits"},
	}
	for in, want := range cases {
		if got := basicTokenize(in); !reflect.DeepEqual(got, want) {
			t.Errorf("basicTokenize(%q) = %q, want %q", in, got, want)
		}
	}
}

// tinyVocab is a vocabulary for WordPiece tests.
var tinyVocab = []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "un", "##aff", "##able", "want", "##ed", "runn", "##ing", ",", "a"}

func TestWordPiece(t *testing.T) {
	tok, err := NewTokenizer(tinyVocab, 16)
	if err != nil {
		t.Fatal(err)
	}
	got := tok.Encode("unaffable, wanted running xyz " + strings.Repeat("a", 101))
	want := []int32{2, 4, 5, 6, 11, 7, 8, 9, 10, 1, 1, 3}
	if !slices.Equal(got, want) {
		t.Fatalf("Encode = %v, want %v", got, want)
	}
	short, _ := NewTokenizer(tinyVocab, 5)
	if got := short.Encode("a a a a a a a"); !slices.Equal(got, []int32{2, 12, 12, 12, 3}) {
		t.Fatalf("truncation: %v", got)
	}
	if _, err := NewTokenizer([]string{"[PAD]", "[UNK]"}, 8); err == nil {
		t.Fatal("vocabulary without [CLS] and [SEP] accepted")
	}
	if _, err := NewTokenizer(tinyVocab, 2); err == nil {
		t.Fatal("maxLen 2 accepted")
	}
}

// writeSafetensors writes F32 tensors (and one I64 tensor) in safetensors format.
func writeSafetensors(t *testing.T, path string, ts map[string]tensor, corrupt func(h map[string]any)) {
	t.Helper()
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var data []byte
	for _, name := range slices.Sorted(func(yield func(string) bool) {
		for k := range ts {
			if !yield(k) {
				return
			}
		}
	}) {
		tt := ts[name]
		start := len(data)
		for _, v := range tt.data {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
		header[name] = map[string]any{"dtype": "F32", "shape": tt.shape, "data_offsets": []int{start, len(data)}}
	}
	header["ids"] = map[string]any{"dtype": "I64", "shape": []int{1}, "data_offsets": []int{len(data), len(data) + 8}}
	data = append(data, make([]byte, 8)...)
	if corrupt != nil {
		corrupt(header)
	}
	hb, _ := json.Marshal(header)
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(hb)))
	out = append(append(out, hb...), data...)
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestReadSafetensors(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "m.safetensors")
	writeSafetensors(t, p, map[string]tensor{"w": {shape: []int{2, 3}, data: []float32{1, 2, 3, 4, 5, 6.5}}}, nil)
	ts, err := readSafetensors(p)
	if err != nil || !reflect.DeepEqual(ts["w"].data, []float32{1, 2, 3, 4, 5, 6.5}) || len(ts) != 1 {
		t.Fatalf("read %v %v", ts, err)
	}
	writeSafetensors(t, p, map[string]tensor{"w": {shape: []int{2, 3}, data: make([]float32, 6)}}, func(h map[string]any) {
		h["w"].(map[string]any)["shape"] = []int{2, 4}
	})
	if _, err := readSafetensors(p); err == nil || !strings.Contains(err.Error(), "offsets") {
		t.Fatalf("shape/offset mismatch: %v", err)
	}
	_ = os.WriteFile(p, binary.LittleEndian.AppendUint64(nil, 1<<40), 0o600)
	if _, err := readSafetensors(p); err == nil {
		t.Fatal("huge header length accepted")
	}
	if _, err := readSafetensors(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("missing file accepted")
	}
}

func TestDotAndLayerNorm(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	a, b := make([]float32, 37), make([]float32, 37)
	var want float64
	for i := range a {
		a[i], b[i] = r.Float32()-0.5, r.Float32()-0.5
		want += float64(a[i]) * float64(b[i])
	}
	if got := dot(a, b); math.Abs(float64(got)-want) > 1e-5 {
		t.Fatalf("dot %v want %v", got, want)
	}
	ln := layerNorm{g: []float32{1, 1, 1, 1}, b: []float32{0, 0, 0, 0}}
	x := []float32{1, 2, 3, 4, 10, 10, 10, 10}
	ln.apply(x, 4, 1e-12)
	var mean, sq float64
	for _, v := range x[:4] {
		mean += float64(v)
		sq += float64(v) * float64(v)
	}
	if math.Abs(mean) > 1e-5 || math.Abs(sq/4-1) > 1e-4 || x[4] != 0 {
		t.Fatalf("layer norm %v", x)
	}
}

func TestAttentionWithEqualScoresAveragesValues(t *testing.T) {
	n, h, heads := 3, 4, 2
	q := make([]float32, n*h) // zero queries: every score equal
	k := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3}
	v := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	ctx := attention(q, k, v, n, h, heads)
	for t2 := range n {
		got := ctx[t2*h : (t2+1)*h]
		want := []float32{5, 6, 7, 8}
		for j := range want {
			if math.Abs(float64(got[j]-want[j])) > 1e-5 {
				t.Fatalf("token %d context %v, want %v", t2, got, want)
			}
		}
	}
}

// tinyModel writes a 1-layer, 4-wide BERT with random weights.
func tinyModel(t *testing.T, corrupt bool) string {
	t.Helper()
	dir := t.TempDir()
	cfg := `{"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":8,"max_position_embeddings":16,"vocab_size":13,"type_vocab_size":2,"layer_norm_eps":1e-12,"hidden_act":"gelu"}`
	_ = os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o600)
	_ = os.WriteFile(filepath.Join(dir, "vocab.txt"), []byte(strings.Join(tinyVocab, "\n")+"\n"), 0o600)
	r := rand.New(rand.NewPCG(3, 4))
	rnd := func(shape ...int) tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		d := make([]float32, n)
		for i := range d {
			d[i] = r.Float32() - 0.5
		}
		return tensor{shape: shape, data: d}
	}
	ts := map[string]tensor{
		"embeddings.word_embeddings.weight":       rnd(13, 4),
		"embeddings.position_embeddings.weight":   rnd(16, 4),
		"embeddings.token_type_embeddings.weight": rnd(2, 4),
		"embeddings.LayerNorm.weight":             rnd(4),
		"embeddings.LayerNorm.bias":               rnd(4),
	}
	p := "encoder.layer.0."
	for name, s := range map[string][2]int{
		"attention.self.query": {4, 4}, "attention.self.key": {4, 4}, "attention.self.value": {4, 4},
		"attention.output.dense": {4, 4}, "intermediate.dense": {8, 4}, "output.dense": {4, 8},
	} {
		ts[p+name+".weight"] = rnd(s[0], s[1])
		ts[p+name+".bias"] = rnd(s[0])
	}
	for _, name := range []string{"attention.output.LayerNorm", "output.LayerNorm"} {
		ts[p+name+".weight"], ts[p+name+".bias"] = rnd(4), rnd(4)
	}
	if corrupt {
		ts[p+"output.dense.weight"] = rnd(8, 4)
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), ts, nil)
	return dir
}

func TestTinyModelEmbedsDeterministically(t *testing.T) {
	m, err := Load(tinyModel(t, false))
	if err != nil {
		t.Fatal(err)
	}
	a, b := m.Embed("unaffable, wanted"), m.Embed("running")
	var norm float64
	for _, v := range a {
		norm += float64(v) * float64(v)
	}
	if m.Dim() != 4 || len(a) != 4 || math.Abs(norm-1) > 1e-5 || slices.Equal(a, b) {
		t.Fatalf("embedding %v (norm %v), other %v", a, norm, b)
	}
	batch := m.EmbedBatch([]string{"unaffable, wanted", "running", "unaffable, wanted"}, 3)
	if !slices.Equal(batch[0], a) || !slices.Equal(batch[1], b) || !slices.Equal(batch[2], a) {
		t.Fatal("EmbedBatch differs from Embed")
	}
	if !slices.Equal(m.Tokens("wanted"), []int32{2, 7, 8, 3}) {
		t.Fatalf("tokens %v", m.Tokens("wanted"))
	}
}

func TestLoadRejectsWrongShapes(t *testing.T) {
	_, err := Load(tinyModel(t, true))
	if err == nil || !strings.Contains(err.Error(), "output.dense.weight has shape [8 4], want [4 8]") {
		t.Fatalf("wrong shape: %v", err)
	}
	if _, err := Load(t.TempDir()); err == nil {
		t.Fatal("empty directory accepted")
	}
}

// modelDir is where make model puts all-MiniLM-L6-v2.
func modelDir() string {
	if d := os.Getenv("HSEARCH_MODEL_DIR"); d != "" {
		return d
	}
	return filepath.Join("..", "..", ".tmp", "models", "all-MiniLM-L6-v2")
}

func TestMatchesTheReferenceEncoder(t *testing.T) {
	dir := modelDir()
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("model not downloaded (run make model): %v", err)
	}
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "minilm-reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Cases []struct {
			Text      string    `json:"text"`
			IDs       []int32   `json:"ids"`
			Embedding []float64 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	worst := 1.0
	for _, c := range ref.Cases {
		if got := m.Tokens(c.Text); !slices.Equal(got, c.IDs) {
			t.Errorf("tokens of %.30q:\n got %v\nwant %v", c.Text, got, c.IDs)
			continue
		}
		e := m.Embed(c.Text)
		var cos, maxDiff float64
		for i, v := range e {
			cos += float64(v) * c.Embedding[i]
			maxDiff = max(maxDiff, math.Abs(float64(v)-c.Embedding[i]))
		}
		worst = min(worst, cos)
		if cos < 0.99999 || maxDiff > 1e-4 {
			t.Errorf("embedding of %.30q: cosine %.7f, max abs diff %.2e", c.Text, cos, maxDiff)
		}
	}
	t.Logf("%d reference cases, lowest cosine %.7f", len(ref.Cases), worst)
}
