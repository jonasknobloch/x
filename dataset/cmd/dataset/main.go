package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/jonasknobloch/x/dataset"
)

func main() {
	root := "./tmp/minipile"

	if err := dataset.Download("JeanKaddour/minipile", "", root); err != nil {
		log.Fatal(err)
	}

	fmt.Println()

	split := filepath.Join(root, "train")

	var reader *dataset.Reader

	if r, err := dataset.NewReader(split); err != nil {
		log.Fatal(err)
	} else {
		reader = r
	}

	i := 0

	for s := range reader.Texts("text") {
		_ = s
		fmt.Printf("\r%d", i+1)
		i++
	}

	if err := reader.Err(); err != nil {
		log.Fatal(err)
	}

	fmt.Println()
}
