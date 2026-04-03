package dataset

import "iter"

type Reader interface {
	Texts() iter.Seq2[int, string]
	Num() (int, error)
	Err() error
}
