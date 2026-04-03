package dataset

import "iter"

type StringReader struct {
	s string
}

func NewStringReader(s string) *StringReader {
	return &StringReader{
		s: s,
	}
}

func (s *StringReader) Texts() iter.Seq2[int, string] {
	return func(yield func(int, string) bool) {
		if !yield(0, s.s) {
			return
		}
	}
}

func (s *StringReader) Err() error {
	return nil
}
