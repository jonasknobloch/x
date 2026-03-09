package llm

type job struct {
	positions []int
	tokens    [][]int64
	seen      []int
	results   []result
	debug     [][2]int
}

func newJob(batchSize int) *job {
	return &job{
		positions: make([]int, 0, batchSize),
		tokens:    make([][]int64, 0, batchSize),
		seen:      make([]int, 0, batchSize),
		results:   make([]result, 0, batchSize),
		debug:     make([][2]int, 0),
	}
}
