package llmc

import (
	"iter"
	"sync"
)

func Pool[In, Out any](seq iter.Seq[In], nWorkers int, f func(In) Out) <-chan Out {
	out := make(chan Out, 256)

	go func() {
		defer close(out)

		type work struct {
			idx int
			in  In
		}

		type result struct {
			idx int
			out Out
		}

		jobs := make(chan work, nWorkers)
		results := make(chan result, nWorkers)

		var wg sync.WaitGroup

		for range nWorkers {
			wg.Add(1)

			go func() {
				defer wg.Done()

				for j := range jobs {
					results <- result{j.idx, f(j.in)}
				}
			}()
		}

		go func() {
			idx := 0

			for item := range seq {
				jobs <- work{idx, item}

				idx++
			}

			close(jobs)

			wg.Wait()

			close(results)
		}()

		next := 0

		pending := make(map[int]Out)

		for r := range results {
			pending[r.idx] = r.out

			for {
				v, ok := pending[next]

				if !ok {
					break
				}

				out <- v

				delete(pending, next)

				next++
			}
		}
	}()

	return out
}
