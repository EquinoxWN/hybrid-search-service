// Package embed turns text into sentence embeddings with a pure-Go BERT encoder. It loads
// sentence-transformers/all-MiniLM-L6-v2 (config.json, vocab.txt, model.safetensors) and
// reproduces its pipeline: uncased WordPiece, 6 transformer layers, mean pooling over tokens,
// then L2 normalisation. No cgo and no model server are needed.
package embed

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
)

type linear struct {
	w       []float32 // [out][in], row-major
	b       []float32
	in, out int
}

type layerNorm struct{ g, b []float32 }

type layer struct {
	q, k, v, attnOut, inter, out linear
	ln1, ln2                     layerNorm
}

type config struct {
	Hidden       int     `json:"hidden_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	Intermediate int     `json:"intermediate_size"`
	MaxPositions int     `json:"max_position_embeddings"`
	Vocab        int     `json:"vocab_size"`
	TypeVocab    int     `json:"type_vocab_size"`
	Eps          float64 `json:"layer_norm_eps"`
	Act          string  `json:"hidden_act"`
}

// Model is a loaded encoder; it is safe for concurrent use.
type Model struct {
	cfg            config
	tok            *Tokenizer
	word, pos, typ []float32
	embLN          layerNorm
	layers         []layer
}

// Load reads config.json, vocab.txt and model.safetensors from dir.
func Load(dir string) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("embed: config.json: %w", err)
	}
	if cfg.Act != "gelu" || cfg.Hidden <= 0 || cfg.Heads <= 0 || cfg.Hidden%cfg.Heads != 0 || cfg.Layers <= 0 || cfg.Layers > 48 {
		return nil, fmt.Errorf("embed: unsupported config %+v", cfg)
	}
	tok, err := LoadVocab(filepath.Join(dir, "vocab.txt"), min(defaultMaxLen, cfg.MaxPositions))
	if err != nil {
		return nil, err
	}
	ts, err := readSafetensors(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	l := loader{ts: ts}
	m := &Model{cfg: cfg, tok: tok}
	h, in := cfg.Hidden, cfg.Intermediate
	m.word = l.get("embeddings.word_embeddings.weight", cfg.Vocab, h)
	m.pos = l.get("embeddings.position_embeddings.weight", cfg.MaxPositions, h)
	m.typ = l.get("embeddings.token_type_embeddings.weight", cfg.TypeVocab, h)
	m.embLN = l.norm("embeddings.LayerNorm", h)
	for i := range cfg.Layers {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		m.layers = append(m.layers, layer{
			q:       l.linear(p+"attention.self.query", h, h),
			k:       l.linear(p+"attention.self.key", h, h),
			v:       l.linear(p+"attention.self.value", h, h),
			attnOut: l.linear(p+"attention.output.dense", h, h),
			ln1:     l.norm(p+"attention.output.LayerNorm", h),
			inter:   l.linear(p+"intermediate.dense", h, in),
			out:     l.linear(p+"output.dense", in, h),
			ln2:     l.norm(p+"output.LayerNorm", h),
		})
	}
	if l.err != nil {
		return nil, l.err
	}
	if len(tok.vocab) > cfg.Vocab {
		return nil, errors.New("embed: vocabulary larger than the embedding table")
	}
	return m, nil
}

type loader struct {
	ts  map[string]tensor
	err error
}

// get returns a tensor's data after checking its shape.
func (l *loader) get(name string, shape ...int) []float32 {
	t, ok := l.ts[name]
	if !ok {
		t, ok = l.ts["bert."+name]
	}
	if !ok {
		l.err = errors.Join(l.err, fmt.Errorf("embed: missing tensor %s", name))
		return nil
	}
	if fmt.Sprint(t.shape) != fmt.Sprint(shape) {
		l.err = errors.Join(l.err, fmt.Errorf("embed: tensor %s has shape %v, want %v", name, t.shape, shape))
		return nil
	}
	return t.data
}

// linear loads a dense layer stored as [out][in].
func (l *loader) linear(name string, in, out int) linear {
	return linear{w: l.get(name+".weight", out, in), b: l.get(name+".bias", out), in: in, out: out}
}

// norm loads a layer norm.
func (l *loader) norm(name string, h int) layerNorm {
	return layerNorm{g: l.get(name+".weight", h), b: l.get(name+".bias", h)}
}

// Dim is the embedding size.
func (m *Model) Dim() int { return m.cfg.Hidden }

// Tokens returns the token IDs the model sees for text.
func (m *Model) Tokens(text string) []int32 { return m.tok.Encode(text) }

// Embed returns the unit-length sentence embedding of text.
func (m *Model) Embed(text string) []float32 { return m.forward(m.tok.Encode(text)) }

// EmbedBatch embeds texts with a fixed number of worker goroutines.
func (m *Model) EmbedBatch(texts []string, workers int) [][]float32 {
	out := make([][]float32, len(texts))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for range max(1, workers) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				out[i] = m.Embed(texts[i])
			}
		}()
	}
	for i := range texts {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return out
}

// forward runs the encoder and mean-pools the last hidden states.
func (m *Model) forward(ids []int32) []float32 {
	n, h := len(ids), m.cfg.Hidden
	x := make([]float32, n*h)
	for t, id := range ids {
		row := x[t*h : (t+1)*h]
		w, p := m.word[int(id)*h:], m.pos[t*h:]
		for j := range row {
			row[j] = w[j] + p[j] + m.typ[j]
		}
	}
	m.embLN.apply(x, h, m.cfg.Eps)
	for i := range m.layers {
		x = m.layers[i].apply(x, n, h, m.cfg.Heads, m.cfg.Eps)
	}
	pooled := make([]float32, h)
	for t := range n {
		for j, v := range x[t*h : (t+1)*h] {
			pooled[j] += v
		}
	}
	var norm2 float64
	for j := range pooled {
		pooled[j] /= float32(n)
		norm2 += float64(pooled[j]) * float64(pooled[j])
	}
	inv := float32(1 / math.Max(math.Sqrt(norm2), 1e-12))
	for j := range pooled {
		pooled[j] *= inv
	}
	return pooled
}

// apply runs one transformer layer on n tokens.
func (l *layer) apply(x []float32, n, h, heads int, eps float64) []float32 {
	q, k, v := l.q.apply(x, n), l.k.apply(x, n), l.v.apply(x, n)
	ctx := attention(q, k, v, n, h, heads)
	a := l.attnOut.apply(ctx, n)
	for i := range a {
		a[i] += x[i]
	}
	l.ln1.apply(a, h, eps)
	f := l.inter.apply(a, n)
	for i, z := range f {
		f[i] = float32(0.5 * float64(z) * (1 + math.Erf(float64(z)/math.Sqrt2)))
	}
	o := l.out.apply(f, n)
	for i := range o {
		o[i] += a[i]
	}
	l.ln2.apply(o, h, eps)
	return o
}

// attention is multi-head scaled dot-product self-attention without masking (one sequence).
func attention(q, k, v []float32, n, h, heads int) []float32 {
	dh := h / heads
	scale := float32(1 / math.Sqrt(float64(dh)))
	ctx := make([]float32, n*h)
	scores := make([]float32, n)
	for hd := range heads {
		off := hd * dh
		for t := range n {
			qt := q[t*h+off : t*h+off+dh]
			maxS := float32(math.Inf(-1))
			for s := range n {
				scores[s] = dot(qt, k[s*h+off:s*h+off+dh]) * scale
				maxS = max(maxS, scores[s])
			}
			var sum float32
			for s := range n {
				scores[s] = float32(math.Exp(float64(scores[s] - maxS)))
				sum += scores[s]
			}
			out := ctx[t*h+off : t*h+off+dh]
			for s := range n {
				p := scores[s] / sum
				vs := v[s*h+off : s*h+off+dh]
				for j := range out {
					out[j] += p * vs[j]
				}
			}
		}
	}
	return ctx
}

// apply computes x W^T + b for n rows.
func (l *linear) apply(x []float32, n int) []float32 {
	y := make([]float32, n*l.out)
	for t := range n {
		xt := x[t*l.in : (t+1)*l.in]
		yt := y[t*l.out : (t+1)*l.out]
		for o := range l.out {
			yt[o] = dot(xt, l.w[o*l.in:(o+1)*l.in]) + l.b[o]
		}
	}
	return y
}

// apply normalises each row of width h in place.
func (ln *layerNorm) apply(x []float32, h int, eps float64) {
	for r := 0; r < len(x); r += h {
		row := x[r : r+h]
		var mean, vr float64
		for _, z := range row {
			mean += float64(z)
		}
		mean /= float64(h)
		for _, z := range row {
			d := float64(z) - mean
			vr += d * d
		}
		inv := 1 / math.Sqrt(vr/float64(h)+eps)
		for j, z := range row {
			row[j] = float32((float64(z)-mean)*inv)*ln.g[j] + ln.b[j]
		}
	}
}

// dot is a 4-way unrolled float32 dot product.
func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}
