package llm

import (
	"context"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/jonasknobloch/mbpe"
	"github.com/jonasknobloch/x/dataset"
)

func (e *Evaluator) Run(data *dataset.Reader, window, stride int) error {
	docs := make([]string, 0)

	n := 0

	for d := range data.Texts("text") {
		if n > 5 {
			break
		}

		docs = append(docs, d)

		n++
	}

	tokens := toInt64(e.tokenizer.Tokenize(strings.Join(docs, "\n\n")))[:10240] // TODO performance

	batchSize := 1

	if len(tokens) < window {
		return nil // TODO handle
	}

	windows := ((len(tokens) - window) / stride) + 1
	jobs := (windows + batchSize - 1) / batchSize

	pb := mbpe.NewProgressBar("Perplexity", 20, jobs, time.Now())

	ctx, cancel := context.WithCancel(context.Background())

	defer cancel()

	done := make(chan struct{})

	go func(ctx context.Context) {
	main:
		for {
			select {
			case <-ctx.Done():
				break main
			default:
				time.Sleep(time.Second * 1)

				e.mutex.RLock()

				j := e.jobs

				e.mutex.RUnlock()

				pb.Update(j)
				pb.Print()

				if j >= jobs {
					break main
				}
			}
		}

		pb.Finish()

		close(done)
	}(ctx)

	if err := e.schedule(tokens, window, stride, batchSize); err != nil {
		log.Fatal(err)
	}

	<-done

	return nil
}

func (e *Evaluator) schedule(tokens []int64, contextSize, stride, batchSize int) error {
	jobs := make(chan *job)

	var wg sync.WaitGroup

	devices := make([]int, len(e.models))

	for i := range len(devices) {
		devices[i] = i
	}

	devicePool := newPool[int](devices...)

	for d := 0; d < devicePool.Len(); d++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := range jobs {
				device := devicePool.Acquire()

				e.execute(j, device)

				devicePool.Release(device)

				e.mutex.Lock()

				for _, p := range j.results {
					e.results = append(e.results, p)
				}

				e.jobs++

				e.mutex.Unlock()
			}
		}()
	}

	j := newJob(batchSize)

	// 0 to 1023: full logits
	// 512 to 1535: 1024 upwards
	// 1024 to 2047: 1536 upwards
	// ...

	seen := 1 // first token as context
	n := 0

	// for i := 0; i < len(tokens); i += stride {
	for i := 0; i+contextSize <= len(tokens); i += stride {
		// if (len(j.positions) == batchSize) || i+stride > len(tokens) {
		if len(j.positions) == batchSize {
			jobs <- j

			j = newJob(batchSize)
		}

		j.positions = append(j.positions, n)
		j.tokens = append(j.tokens, tokens[i:min(i+contextSize, len(tokens))]) // TODO verify
		j.seen = append(j.seen, seen-i)

		seen = i + contextSize

		n++
	}

	if len(j.positions) > 0 {
		jobs <- j
	}

	close(jobs)

	wg.Wait()

	return nil
}

func (e *Evaluator) execute(j *job, device int) {
	if len(j.positions) != 1 {
		panic("unimplemented")
	}

	if j.seen[0] < 1 {
		panic("empty context")
	}

	m := e.models[device]

	logits := make([][]float32, 0, len(j.tokens[0]))

	// fmt.Println("executing job", j.positions[0])

	if _, err := m.Generate(j.tokens[0], 0, &logits); err != nil {
		panic(err) // TODO handle
	}

	eLogits := logits[j.seen[0]-1 : len(logits)-1]
	eTargets := toInt(j.tokens[0][j.seen[0]:])

	v := e.Execute(eLogits, eTargets)

	j.results = append(j.results, result{
		value: v,
		n:     len(eTargets),
	})

	return
}
