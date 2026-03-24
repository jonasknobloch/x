package llm

import (
	"fmt"
	"sync"
	"time"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/tui"
)

func (e *Evaluator[R]) Run(data dataset.Reader, window, stride int) error {
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

	pb := tui.NewProgressBar("Perplexity", 20, 0, time.Now())

	pb.Start(1*time.Second, func() int {
		return int(e.completed.Load())
	})

	defer pb.Close()

	n := 0

	for d := range data.Texts("text") {
		tokens := toInt64(e.tokenizer.Tokenize(d))

		e.schedule(n, tokens, window, stride, e.batchSize)

		pb.SetTotal(pb.Total() + e.estimateJobs(tokens, window, stride))

		n++
	}

	close(e.jobs)

	wg.Wait()

	close(e.results)

	return nil
}

func (e *Evaluator[R]) estimateJobs(tokens []int64, window, stride int) int {
	if len(tokens) < window {
		return 0
	}

	windows := ((len(tokens) - window) / stride) + 1
	// jobs := (windows + batchSize - 1) / e.batchSize

	return windows
}

func (e *Evaluator[R]) schedule(uid int, tokens []int64, contextSize, stride, batchSize int) {
	b := newBatch(batchSize)

	seen := 1 // first token as context
	n := 0

	// for i := 0; i+contextSize <= len(tokens); i += stride {
	for i := 0; i < len(tokens); i += stride {
		if b.Size() == batchSize {
			e.jobs <- *b

			b = newBatch(batchSize)
		}

		j := min(i+contextSize, len(tokens))

		if j-i < contextSize {
			break // don't add jobs with partial windows
		}

		// if j < seen {
		// 	break // don't add jobs with no new tokens
		// }

		b.AddJob(Job{
			Document: uid,
			Position: n,
			Tokens:   tokens[i:j],
			Seen:     seen - i,
		})

		seen = i + contextSize

		n++
	}

	if b.Size() > 0 {
		e.jobs <- *b
	}

	e.scheduled.Add(int64(n))
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
