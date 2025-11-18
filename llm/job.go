package llm

type job struct {
	positions []int
	tokens    [][]int64
	seen      []int
	results   []float64
	debug     [][2]int
}

func newJob(batchSize int) *job {
	return &job{
		positions: make([]int, 0, batchSize),
		tokens:    make([][]int64, 0, batchSize),
		seen:      make([]int, 0, batchSize),
		results:   make([]float64, 0, batchSize),
		debug:     make([][2]int, 0),
	}
}
