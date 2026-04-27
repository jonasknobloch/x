package dataset

import (
	"bufio"
	"os"
	"sync"
	"testing"

	"go.jknobloc.com/x/shelf"
)

func TestCountLinesAll(t *testing.T) {
	names := []string{
		shelf.Abs("data/babylm/train_100M/bnc_spoken.train"),
		shelf.Abs("data/babylm/train_100M/childes.train"),
		shelf.Abs("data/babylm/train_100M/gutenberg.train"),
		shelf.Abs("data/babylm/train_100M/open_subtitles.train"),
		shelf.Abs("data/babylm/train_100M/simple_wiki.train"),
		shelf.Abs("data/babylm/train_100M/switchboard.train"),
	}

	n, err := countLinesAll(names, []byte("\n"))

	if err != nil {
		t.Fatal(err)
	}

	m, err := countLinesNaive(names...)

	if err != nil {
		t.Fatal(err)
	}

	if n != m {
		t.Errorf("expected %d but got %d\n", m, n)
	}
}

func BenchmarkCountAll(b *testing.B) {
	names := []string{
		shelf.Abs("data/babylm/train_100M/bnc_spoken.train"),
		shelf.Abs("data/babylm/train_100M/childes.train"),
		shelf.Abs("data/babylm/train_100M/gutenberg.train"),
		shelf.Abs("data/babylm/train_100M/open_subtitles.train"),
		shelf.Abs("data/babylm/train_100M/simple_wiki.train"),
		shelf.Abs("data/babylm/train_100M/switchboard.train"),
	}

	for i := 0; i < b.N; i++ {
		_, err := countLinesAll(names, []byte("\n"))

		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCountLinesNaive(b *testing.B) {
	names := []string{
		shelf.Abs("data/babylm/train_100M/bnc_spoken.train"),
		shelf.Abs("data/babylm/train_100M/childes.train"),
		shelf.Abs("data/babylm/train_100M/gutenberg.train"),
		shelf.Abs("data/babylm/train_100M/open_subtitles.train"),
		shelf.Abs("data/babylm/train_100M/simple_wiki.train"),
		shelf.Abs("data/babylm/train_100M/switchboard.train"),
	}

	for i := 0; i < b.N; i++ {
		_, err := countLinesNaive(names...)

		if err != nil {
			b.Fatal(err)
		}
	}
}

func countLinesNaive(names ...string) (int, error) {
	var wg sync.WaitGroup

	results := make(chan int, len(names))
	errors := make(chan error, len(names))

	for _, name := range names {
		wg.Add(1)

		go func() {
			defer wg.Done()

			var scanner *bufio.Scanner

			if file, err := os.Open(name); err != nil {
				results <- 0
				errors <- err

				return
			} else {
				scanner = bufio.NewScanner(file)

				buf := make([]byte, 0, 1024*1024)

				scanner.Buffer(buf, 1024*1024)

				defer file.Close()
			}

			count := 0

			for scanner.Scan() {
				count++
			}

			if err := scanner.Err(); err != nil {
				results <- 0
				errors <- err

				return
			}

			results <- count
		}()
	}

	wg.Wait()

	close(results)
	close(errors)

	total := 0

	for count := range results {
		total += count
	}

	return total, <-errors
}
