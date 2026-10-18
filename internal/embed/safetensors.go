package embed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// tensor is a dense float32 array with its shape.
type tensor struct {
	shape []int
	data  []float32
}

const (
	maxHeader       = 16 << 20
	maxWeightsBytes = 1 << 30
)

// readSafetensors loads every F32 tensor of a .safetensors file, checking all offsets.
func readSafetensors(path string) (map[string]tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	if size < 8 || size > maxWeightsBytes {
		return nil, fmt.Errorf("embed: %s: size %d out of range", path, size)
	}
	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, err
	}
	n := binary.LittleEndian.Uint64(lenBuf[:])
	if n == 0 || n > maxHeader || int64(n) > size-8 {
		return nil, fmt.Errorf("embed: %s: bad header length %d", path, n)
	}
	hdr := make([]byte, n)
	if _, err := io.ReadFull(f, hdr); err != nil {
		return nil, err
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(hdr, &entries); err != nil {
		return nil, fmt.Errorf("embed: %s: header: %w", path, err)
	}
	data := make([]byte, size-8-int64(n))
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, err
	}
	out := map[string]tensor{}
	for name, raw := range entries {
		if name == "__metadata__" {
			continue
		}
		var e struct {
			Dtype   string   `json:"dtype"`
			Shape   []int    `json:"shape"`
			Offsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &e); err != nil {
			return nil, fmt.Errorf("embed: tensor %s: %w", name, err)
		}
		if e.Dtype != "F32" {
			continue
		}
		count := 1
		for _, d := range e.Shape {
			if d <= 0 || d > 1<<24 || count > math.MaxInt32/d {
				return nil, fmt.Errorf("embed: tensor %s: bad shape %v", name, e.Shape)
			}
			count *= d
		}
		b, end := e.Offsets[0], e.Offsets[1]
		if b < 0 || end > int64(len(data)) || end-b != int64(4*count) {
			return nil, fmt.Errorf("embed: tensor %s: offsets %v do not match shape %v", name, e.Offsets, e.Shape)
		}
		vals := make([]float32, count)
		for i := range vals {
			vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[b+int64(4*i):]))
		}
		out[name] = tensor{shape: e.Shape, data: vals}
	}
	return out, nil
}
