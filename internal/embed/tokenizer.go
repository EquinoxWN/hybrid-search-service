package embed

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

const (
	maxVocabFile    = 4 << 20
	maxWordChars    = 100
	maxTextBytes    = 64 << 10
	defaultMaxLen   = 256
	unknownToken    = "[UNK]"
	classifierToken = "[CLS]"
	separatorToken  = "[SEP]"
)

// Tokenizer is the uncased BERT tokenizer: basic splitting, then greedy WordPiece.
type Tokenizer struct {
	vocab         map[string]int32
	unk, cls, sep int32
	maxLen        int
}

// NewTokenizer builds a tokenizer from vocabulary entries in ID order.
func NewTokenizer(tokens []string, maxLen int) (*Tokenizer, error) {
	if maxLen < 3 {
		return nil, errors.New("embed: maxLen must be at least 3")
	}
	t := &Tokenizer{vocab: make(map[string]int32, len(tokens)), maxLen: maxLen}
	for i, tok := range tokens {
		if _, dup := t.vocab[tok]; !dup {
			t.vocab[tok] = int32(i)
		}
	}
	var ok [3]bool
	t.unk, ok[0] = t.vocab[unknownToken]
	t.cls, ok[1] = t.vocab[classifierToken]
	t.sep, ok[2] = t.vocab[separatorToken]
	if !ok[0] || !ok[1] || !ok[2] {
		return nil, errors.New("embed: vocabulary lacks [UNK], [CLS] or [SEP]")
	}
	return t, nil
}

// LoadVocab reads a vocab.txt file (one token per line, line number = ID).
func LoadVocab(path string, maxLen int) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if st, err := f.Stat(); err != nil || st.Size() > maxVocabFile {
		return nil, fmt.Errorf("embed: vocabulary %s missing or larger than %d bytes", path, maxVocabFile)
	}
	var tokens []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		tokens = append(tokens, strings.TrimRight(sc.Text(), "\r"))
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return NewTokenizer(tokens, maxLen)
}

// Encode returns [CLS] + word pieces + [SEP], truncated to the maximum length.
func (t *Tokenizer) Encode(text string) []int32 {
	if len(text) > maxTextBytes {
		text = text[:maxTextBytes]
	}
	ids := []int32{t.cls}
	for _, w := range basicTokenize(text) {
		ids = append(ids, t.wordPiece(w)...)
		if len(ids) >= t.maxLen-1 {
			ids = ids[:t.maxLen-1]
			break
		}
	}
	return append(ids, t.sep)
}

// wordPiece splits one word by greedy longest-match against the vocabulary.
func (t *Tokenizer) wordPiece(word string) []int32 {
	runes := []rune(word)
	if len(runes) > maxWordChars {
		return []int32{t.unk}
	}
	var out []int32
	for start := 0; start < len(runes); {
		end, id := len(runes), int32(-1)
		for ; start < end; end-- {
			piece := string(runes[start:end])
			if start > 0 {
				piece = "##" + piece
			}
			if v, ok := t.vocab[piece]; ok {
				id = v
				break
			}
		}
		if id < 0 {
			return []int32{t.unk}
		}
		out = append(out, id)
		start = end
	}
	return out
}

// basicTokenize cleans, lower-cases, strips accents and splits on whitespace and punctuation.
func basicTokenize(text string) []string {
	var b strings.Builder
	for _, r := range text {
		switch {
		case r == 0 || r == unicode.ReplacementChar || isControl(r):
		case isWhitespace(r):
			b.WriteByte(' ')
		case isCJK(r):
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
		default:
			b.WriteRune(r)
		}
	}
	var out []string
	for _, tok := range strings.Fields(b.String()) {
		tok = stripAccents(strings.ToLower(tok))
		start := 0
		rs := []rune(tok)
		for i, r := range rs {
			if isPunct(r) {
				if start < i {
					out = append(out, string(rs[start:i]))
				}
				out = append(out, string(r))
				start = i + 1
			}
		}
		if start < len(rs) {
			out = append(out, string(rs[start:]))
		}
	}
	return out
}

// stripAccents removes combining marks after canonical decomposition.
func stripAccents(s string) string {
	var b strings.Builder
	for _, r := range norm.NFD.String(s) {
		if !unicode.Is(unicode.Mn, r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// isWhitespace matches BERT's whitespace set.
func isWhitespace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' || unicode.Is(unicode.Zs, r)
}

// isControl matches BERT's control set: categories C*, except tab and newlines.
func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.Is(unicode.C, r) || !unicode.In(r, unicode.L, unicode.M, unicode.N, unicode.P, unicode.S, unicode.Z)
}

// isPunct matches BERT's punctuation set: ASCII symbols and categories P*.
func isPunct(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.Is(unicode.P, r)
}

// isCJK reports whether r is a CJK ideograph, which BERT tokenizes one character at a time.
func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) || (r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) || (r >= 0x2B740 && r <= 0x2B81F) || (r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) || (r >= 0x2F800 && r <= 0x2FA1F)
}
