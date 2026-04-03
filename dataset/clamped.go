package dataset

import "iter"

type ClampedReader struct {
	Reader
	n int
}

func NewClampedReader(r Reader, n int) *ClampedReader {
	return &ClampedReader{
		Reader: r,
		n:      n,
	}
}

func (c *ClampedReader) Num() (int, error) {
	n, err := c.Reader.Num()

	if err != nil {
		return 0, err
	}

	return min(n, c.n), nil
}

func (c *ClampedReader) Texts() iter.Seq2[int, string] {
	return func(yield func(int, string) bool) {
		for n, d := range c.Reader.Texts() {
			if n >= c.n {
				return
			}

			if !yield(n, d) {
				return
			}
		}
	}
}
