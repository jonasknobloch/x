package main

import (
	"flag"
	"log"
	"os"

	"go.jknobloc.com/x/shelf"
)

func main() {
	rootPath := flag.String("root", ".shelf", "Path to the shelf")
	ignorePath := flag.String("ignore", ".shelfignore", "Path to .shelfignore")

	flag.Parse()

	shelf.Root = *rootPath

	ignore, _ := os.ReadFile(*ignorePath)

	var sum shelf.Sum

	if s, err := shelf.Index(ignore); err != nil {
		log.Fatal(err)
	} else {
		sum = s
	}

	if err := shelf.Serialize(sum, os.Stdout); err != nil {
		log.Fatal(err)
	}
}
