package matrix

import "github.com/jonasknobloch/x/tensor"

type Matrix[T any] struct {
	tensor.Dense[T]
}

// TODO .ToMatrix() helper which converts dense into matrix; panic if rank != 2
// TODO .ToVector() in similar fashion
// TODO Matrix(dense Dense[T]) Dense[T] to not add to dense struct?
// TODO Vector(dense Dense[T]) Dense[T]

func New[T any](rows, cols int) Matrix[T] {
	return Matrix[T]{tensor.NewDense[T]([]int{rows, cols})}
}

func (m Matrix[T]) Rows(row []T) func(func(int, []T) bool) {
	r := m.Rank()

	shape := m.Shape()

	rows := shape[0]

	return func(yield func(int, []T) bool) {
		buffer := make([]int, r-1)

		for i := range rows {

			for idxs, v := range m.Select(0, i).All(buffer) {
				col := idxs[0]

				row[col] = v
			}

			if !yield(i, row) {
				return
			}
		}
	}
}
