package llm

import (
	"bufio"
	"context"
	"log"
	"math"
	mbpe "mbpe-dyn"
	"slices"
	"sync"
	"time"
)

func (e *Evaluator) Perplexity(name string) (float64, error) {
	tokens := make([]int64, 0)

	if err := mbpe.FromFile(name, func(scanner *bufio.Scanner) error {
		for scanner.Scan() {
			line := scanner.Text()

			if err := scanner.Err(); err != nil {
				return err
			}

			line += "\n"

			tokens = append(tokens, ToInt64(e.tokenizer.Tokenize(line))...)
		}

		return nil
	}); err != nil {
		return 0, err
	}

	contextSize, stride, batchSize := 1024, 512, 1

	if len(tokens) < contextSize {
		return 0, nil // TODO handle
	}

	windows := ((len(tokens) - contextSize) / stride) + 1
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

	if err := e.schedule(tokens, contextSize, stride, batchSize); err != nil {
		log.Fatal(err)
	}

	<-done

	total := float64(0)

	for _, nll := range e.results {
		total += nll
	}

	average := total / float64(len(e.results)) * float64(contextSize-1)

	return math.Exp(total / average), nil
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

	seen := 0
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

	m := e.models[device]

	logits := make([][]float32, 0, len(j.tokens[0]))

	// fmt.Println("executing job", j.positions[0])

	if _, err := m.Generate(j.tokens[0], 0, &logits); err != nil {
		panic(err) // TODO handle
	}

	nll := NegLogLikelihood(logits[j.seen[0]:len(logits)-1], ToInt(j.tokens[0][1+j.seen[0]:]))

	j.results = append(j.results, nll)

	return
}

func NegLogLikelihood(logits [][]float32, targets []int) float64 {
	if len(logits) != len(targets) {
		panic("mismatched input lengths")
	}

	total := float64(0)

	for i, target := range targets {
		maxLogit := float64(slices.Max(logits[i]))

		sumExp := float64(0)

		for _, v := range logits[i] {
			sumExp += math.Exp(float64(v) - maxLogit)
		}

		logSumExp := maxLogit + math.Log(sumExp)

		targetLogit := float64(logits[i][target])

		logProb := targetLogit - logSumExp

		total -= logProb
	}

	return total
}
