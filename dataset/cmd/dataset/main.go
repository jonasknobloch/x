package main

import (
	"fmt"
	"log"
	"path/filepath"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/shelf"
)

func main() {
	root := shelf.Abs("data/minipile")

	if err := dataset.Download("JeanKaddour/minipile", "", root); err != nil {
		log.Fatal(err)
	}

	fmt.Println()

	split := filepath.Join(root, "train")

	var reader *dataset.ParquetReader

	if r, err := dataset.NewParquetReader(split); err != nil {
		log.Fatal(err)
	} else {
		reader = r
	}

	for n := range reader.Texts() {
		fmt.Printf("\r%d", n)
	}

	if err := reader.Err(); err != nil {
		log.Fatal(err)
	}

	fmt.Println()
}
