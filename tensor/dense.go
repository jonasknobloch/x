package tensor

import (
	"slices"
)

type Dense[T any] struct {
	base    []T
	offset  int
	shape   []int
	strides []int
}

func NewDense[T any](shape []int) Dense[T] {
	r := len(shape)

	if r == 0 {
		return Dense[T]{
			base:    make([]T, 1),
			offset:  0,
			shape:   make([]int, 0),
			strides: make([]int, 0),
		}
	}

	n := 1

	for dim := range slices.Backward(shape) {
		if shape[dim] < 0 {
			panic("negative size")
		}

		n *= shape[dim]
	}

	strides := make([]int, len(shape))

	for dim := range slices.Backward(strides) {
		if dim == r-1 {
			strides[dim] = 1

			continue
		}

		strides[dim] = strides[dim+1] * shape[dim+1]
	}

	out := Dense[T]{
		base:    make([]T, n),
		offset:  0,
		shape:   make([]int, len(shape)),
		strides: strides,
	}

	copy(out.shape, shape)

	return out
}

func (d Dense[T]) Shape() []int {
	shape := make([]int, len(d.shape))

	copy(shape, d.shape)

	return shape
}

func (d Dense[T]) Size() int {
	n := 1

	for _, size := range d.shape {
		n *= size
	}

	return n
}

func (d Dense[T]) Rank() int {
	return len(d.shape)
}

func (d Dense[T]) Set(idxs []int, val T) {
	storage := d.storageIndex(idxs)

	d.base[storage] = val
}

func (d Dense[T]) At(idxs []int) T {
	storage := d.storageIndex(idxs)

	return d.base[storage]
}

func (d Dense[T]) IsEmpty() bool {
	return d.Size() == 0
}

func (d Dense[T]) IsScalar() bool {
	return d.Rank() == 0
}

func (d Dense[T]) IsContiguous() bool {
	p := 1

	for dim, stride := range slices.Backward(d.strides) {
		if p != stride {
			return false
		}

		p *= d.shape[dim]
	}

	return true
}

func (d Dense[T]) Contiguous() Dense[T] {
	if d.IsContiguous() {
		out := NewDense[T](d.shape)

		copy(out.base, d.base[d.offset:d.offset+d.Size()])

		return out
	}

	out := NewDense[T](d.shape)

	idxs := make([]int, d.Rank())

	linear := 0

	for _, v := range d.All(idxs) {
		out.base[linear] = v

		linear++
	}

	return out
}

func (d Dense[T]) Permute(dims []int) Dense[T] {
	r := d.Rank()

	if len(dims) != r {
		panic("invalid dimensions")
	}

	seen := make([]bool, r)

	shape := make([]int, r)
	strides := make([]int, r)

	for out, in := range dims {
		if in < 0 || in >= r {
			panic("invalid dimension")
		}

		if seen[in] {
			panic("duplicate dimension")
		}

		seen[in] = true

		shape[out] = d.shape[in]
		strides[out] = d.strides[in]
	}

	return Dense[T]{
		base:    d.base,
		offset:  d.offset,
		shape:   shape,
		strides: strides,
	}
}

func (d Dense[T]) Slice(dim, start, end int) Dense[T] {
	r := d.Rank()

	if dim < 0 || dim >= r {
		panic("invalid dimension")
	}

	if start < 0 || start > end || end > d.shape[dim] {
		panic("invalid interval")
	}

	shape := make([]int, r)
	strides := make([]int, r)

	copy(shape, d.shape)
	copy(strides, d.strides)

	shape[dim] = end - start

	return Dense[T]{
		base:    d.base,
		offset:  d.offset + start*strides[dim],
		shape:   shape,
		strides: strides,
	}
}

func (d Dense[T]) Select(dim, index int) Dense[T] {
	r := d.Rank()

	if dim < 0 || dim >= r {
		panic("invalid dimension")
	}

	if index < 0 || index >= d.shape[dim] {
		panic("invalid index")
	}

	shape := make([]int, 0, len(d.shape)-1)
	strides := make([]int, 0, len(d.strides)-1)

	for dimension := range r {
		if dimension == dim {
			continue
		}

		shape = append(shape, d.shape[dimension])
		strides = append(strides, d.strides[dimension])
	}

	return Dense[T]{
		base:    d.base,
		offset:  d.offset + index*d.strides[dim],
		shape:   shape,
		strides: strides,
	}
}

func (d Dense[T]) All(idxs []int) func(func([]int, T) bool) {
	r := d.Rank()
	n := d.Size()

	if len(idxs) != r {
		panic("invalid indices")
	}

	return func(yield func([]int, T) bool) {
		for i := range idxs {
			idxs[i] = 0
		}

		storage := d.offset

		for range n {
			if !yield(idxs, d.base[storage]) {
				return
			}

			for dim, size := range slices.Backward(d.shape) {
				idxs[dim]++
				storage += d.strides[dim]

				if idxs[dim] < size {
					break
				}

				idxs[dim] = 0
				storage -= size * d.strides[dim]
			}
		}
	}
}

func (d Dense[T]) unravel(linear int, idxs []int) {
	r := d.Rank()
	n := d.Size()

	if len(idxs) != r {
		panic("invalid indices")
	}

	if linear < 0 || linear >= n {
		panic("invalid linear index")
	}

	for dim, size := range slices.Backward(d.shape) {
		idxs[dim] = linear % size

		linear /= size
	}
}

func (d Dense[T]) ravel(idxs []int) int {
	r := d.Rank()

	if len(idxs) != r {
		panic("invalid indices")
	}

	linear := 0

	p := 1

	for dim, size := range slices.Backward(d.shape) {
		if idxs[dim] < 0 || idxs[dim] >= size {
			panic("invalid index")
		}

		linear += idxs[dim] * p

		p *= size
	}

	return linear
}

func (d Dense[T]) storageIndex(idxs []int) int {
	r := d.Rank()

	if len(idxs) != r {
		panic("invalid indices")
	}

	storage := d.offset

	for dim := range r {
		if idxs[dim] < 0 || idxs[dim] >= d.shape[dim] {
			panic("invalid index")
		}

		storage += idxs[dim] * d.strides[dim]
	}

	return storage
}

func (d Dense[T]) Data() ([]T, bool) {
	if !d.IsContiguous() {
		return nil, false
	}

	n := d.Size()

	return d.base[d.offset : d.offset+n], true
}
