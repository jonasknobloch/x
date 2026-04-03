package dataset

import "iter"

type Reader interface {
	Texts() iter.Seq[string]
	Err() error
}
