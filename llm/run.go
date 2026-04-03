package llm

import (
	"fmt"
	"sync"
	"time"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/tui"
)

func (e *Evaluator[R]) Run(title string, data dataset.Reader, window, stride int) error {
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

					defer func() {
						if r := recover(); r != nil {
							fmt.Println("HOUSTON") // TODO handle
						}
					}()

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

	tb := NewTokenBuffer(e.tokenizer, window, stride)

	tb.SetIncludeTail(false)

	b := newBatch(e.batchSize)

	doc := 0
	pos := 0

	for n, d := range data.Texts() {
		if n != doc {
			pos = 0
			doc = n
		}

		for w, s := range tb.Push(n, d) {
			if s == 0 {
				s = 1 // first token as context
			}

			b.AddJob(Job{
				Document: doc,
				Position: pos,
				Tokens:   w,
				Seen:     s,
			})

			pos++

			if b.Size() == e.batchSize {
				e.jobs <- *b

				b = newBatch(e.batchSize)

				e.scheduled.Add(int64(e.batchSize))

				pb.SetTotal(int(e.scheduled.Load()))
			}
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
	if j.Size() != 1 {
		panic("unimplemented")
	}

	job := j.jobs[0]

	if job.Seen < 1 {
		panic("empty context")
	}

	m := e.models[device]

	logits := make([][]float32, 0, len(job.Tokens))

	if _, err := m.Generate(job.Tokens, 0, &logits); err != nil {
		panic(err) // TODO handle
	}

	l := logits[job.Seen-1 : len(logits)-1]
	t := toInt(job.Tokens[job.Seen:])

	r := e.callback(job, l, t)

	e.results <- r

	return
}
