package tensor

type Contiguous[T any] struct {
	base  []T
	shape []int
}

type ContiguousF32 Contiguous[float32]

func NewContiguous[T any](shape []int) *Contiguous[T] {
	n := 1

	for _, s := range shape {
		if s < 0 {
			panic("negative dimension")
		}

		n *= s
	}

	return &Contiguous[T]{
		base:  make([]T, n),
		shape: shape,
	}
}

func (c *Contiguous[T]) Permute(axes []int) *Contiguous[T] {
	out := &Contiguous[T]{
		base:  make([]T, len(c.base)),
		shape: make([]int, len(c.shape)),
	}

	if len(axes) != len(out.shape) {
		// TODO panic
	}

	seen := make(map[int]struct{})

	for i, a := range axes {
		if a < 0 {
			// TODO panic
		}

		if a > len(out.shape)-1 {
			// TODO panic
		}

		if _, ok := seen[a]; ok {
			// TODO panic
		}

		out.shape[i] = c.shape[a]
		seen[a] = struct{}{}
	}

	stridesIn := strides(c.shape)
	// stridesOut := strides(out.shape)

	rank := len(c.shape)

	outIdx := make([]int, rank)
	inIdx := make([]int, rank)

	for o := range len(out.base) {
		unravel(o, out.shape, outIdx)

		for d := range rank {
			inIdx[axes[d]] = outIdx[d]
		}

		// TODO fuse ravel and assignment

		i := ravel(inIdx, stridesIn)

		out.base[o] = c.base[i]
	}

	return out
}

// can i give an iterator instead?

// row-major + contiguous
func strides(shape []int) []int {
	rank := len(shape)

	r := make([]int, rank)

	s := 1

	for i := rank - 1; i >= 0; i-- {
		r[i] = s

		s *= shape[i]
	}

	return r
}

// linear to multi / unflatten
// row major!
func unravel(linear int, shape []int, idx []int) {
	if len(shape) != len(idx) {
		// TODO panic
	}

	for d := len(shape) - 1; d >= 0; d-- {
		s := shape[d]

		// TODO len(out.base)==0.

		idx[d] = linear % s
		linear /= s
	}
}

func ravel(idx, strides []int) int {
	if len(idx) != len(strides) {
		// TODO panic
	}

	l := 0

	for d := range idx {
		l += idx[d] * strides[d]
	}

	return l
}

// dim dimension index
// size number of elements in that dim
// rank number of dims

// column-major + contiguous
// func strides2(shape []int) []int {
// 	rank := len(shape)
//
// 	r := make([]int, rank)
//
// 	s := 1
//
// 	for i := 0; i < rank; i++ {
// 		r[i] = s
//
// 		s *= shape[i]
// 	}
// 	return r
// }

// ravel
// unravel

// odometer for non contiguous tensors?
