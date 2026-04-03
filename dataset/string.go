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

func (s *StringReader) Texts() iter.Seq[string] {
	return func(yield func(string) bool) {
		if !yield(s.s) {
			return
		}
	}
}

func (s *StringReader) Err() error {
	return nil
}
