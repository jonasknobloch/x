package dataset

import "iter"

type Reader interface {
	Texts(column string) iter.Seq[string]
	Err() error
}
