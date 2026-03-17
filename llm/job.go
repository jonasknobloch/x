package llm

type Job struct {
	Document int
	Position int
	Tokens   []int64
	Seen     int
}

type batch struct {
	jobs []Job
}

func newBatch(capacity int) *batch {
	return &batch{
		jobs: make([]Job, 0, capacity),
	}
}

func (b *batch) Size() int {
	return len(b.jobs)
}

func (b *batch) AddJob(job Job) {
	if b.Size() == cap(b.jobs) {
		panic("batch full")
	}

	b.jobs = append(b.jobs, job)
}
