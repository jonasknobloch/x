package llm

import (
	"sync"
	"time"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/tui"
)

func (e *Evaluator[R]) Run(title string, data dataset.Reader, cfg TokenBufferConfig) error {
	devices := make([]int, len(e.models))

	for i := range len(devices) {
		devices[i] = i
	}

	devicePool := newPool(devices...)

	var wg sync.WaitGroup

	for range e.numWorkers {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for b := range e.jobs {
				device := devicePool.Acquire()

				func() {
					defer devicePool.Release(device)

					e.execute(&b, device)
				}()

				e.completed.Add(int64(b.Size()))
			}
		}()
	}

	pb := tui.NewProgressBar(title, 20, 0, time.Now())

	pb.Start(1*time.Second, func() int {
		return int(e.completed.Load())
	})

	defer pb.Close()

	tb := NewTokenBuffer(e.tokenizer, cfg)

	tb.SetIncludeTail(cfg.PadLeft || cfg.PadRight)

	b := newBatch(e.batchSize)

	doc := 0
	pos := 0

	for w := range tb.Stream(data.Texts()) {
		n := w.Document

		if n != doc {
			pos = 0
			doc = n
		}

		tokens := w.Tokens
		seen := w.Seen

		if seen == 0 {
			seen = 1 // first token as context
		}

		b.AddJob(Job{
			Document:     doc,
			Position:     pos,
			Tokens:       tokens,
			Seen:         seen,
			PaddingLeft:  w.PaddingLeft,
			PaddingRight: w.PaddingRight,
		})

		pos++

		if b.Size() == e.batchSize {
			e.jobs <- *b

			b = newBatch(e.batchSize)

			e.scheduled.Add(int64(e.batchSize))

			pb.SetTotal(int(e.scheduled.Load()))
		}
	}

	if s := b.Size(); s > 0 {
		e.jobs <- *b

		e.scheduled.Add(int64(s))

		pb.SetTotal(int(e.scheduled.Load()))
	}

	close(e.jobs)

	wg.Wait()

	close(e.results)

	return nil
}

func (e *Evaluator[R]) execute(j *batch, device int) {
	b := j.Size()

	s := len(j.jobs[0].Tokens)

	tokens := make([]int64, b*s)

	for i, job := range j.jobs {
		copy(tokens[i*s:], job.Tokens)
	}

	m := e.models[device]

	logProbs := make([]float32, 0, b*(s-1))

	if err := m.Score(tokens, b, &logProbs); err != nil {
		panic(err) // TODO handle
	}

	for i, job := range j.jobs {
		if job.Seen < 1 {
			panic("empty context")
		}

		l := logProbs[i*(s-1)+job.PaddingLeft+job.Seen-1 : (i+1)*(s-1)]
		t := toInt(job.Tokens[job.PaddingLeft+job.Seen:])

		if job.PaddingRight > 0 {
			l = l[:len(l)-job.PaddingRight]
			t = t[:len(t)-job.PaddingRight]
		}

		r := e.callback(job, l, t)

		e.results <- r
	}
}
