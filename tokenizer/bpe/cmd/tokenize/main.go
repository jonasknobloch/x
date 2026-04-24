package main

import (
	"flag"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"sync/atomic"
	"time"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llm"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tokenizer/bpe"
	"go.jknobloc.com/x/tui"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")
var memprofile = flag.String("memprofile", "", "write memory profile to `file`")

func main() {
	flag.Parse()

	if *cpuprofile != "" {
		var file *os.File

		if f, err := os.Create(*cpuprofile); err != nil {
			log.Fatal("could not create CPU profile: ", err)
		} else {
			file = f

			defer file.Close()
		}

		if err := pprof.StartCPUProfile(file); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}

		defer pprof.StopCPUProfile()
	}

	reader := data()

	t := tokenizer()

	pb := tui.NewProgressBar("Tokenize", 20, 1000, time.Now())

	var processed atomic.Int64

	pb.Start(1*time.Second, func() int {
		return int(processed.Load())
	})

	defer pb.Close()

	for n, d := range reader.Texts() {
		if n >= 1000 {
			break
		}

		tokens := t.Tokenize(d)

		_ = tokens

		processed.Add(1)
	}

	if *memprofile != "" {
		var file *os.File

		if f, err := os.Create(*memprofile); err != nil {
			log.Fatal("could not create memory profile: ", err)
		} else {
			file = f

			defer file.Close()
		}

		runtime.GC()

		if err := pprof.Lookup("allocs").WriteTo(file, 0); err != nil {
			log.Fatal("could not write memory profile: ", err)
		}
	}
}

func data() *dataset.ParquetReader {
	var simple *dataset.ParquetReader

	if r, err := dataset.NewParquetReader(shelf.Abs("data/wikipedia/20231101/simple/train")); err != nil {
		log.Fatal(err)
	} else {
		simple = r
	}

	return simple
}

func tokenizer() llm.Tokenizer {
	var tok *bpe.Tokenizer

	if t, err := bpe.NewTokenizerFromFiles(shelf.Abs("models/gpt2/vocab.json"), shelf.Abs("models/gpt2/merges.txt")); err != nil {
		log.Fatal(err)
	} else {
		tok = t
	}

	_ = bpe.ByteCoverage(tok)

	return tok
}
