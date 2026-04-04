package main

import (
	"fmt"
	"log"

	"go.jknobloc.com/x/gpt2"
)

func main() {
	prompt := []int64{464, 2068, 7586, 21831}

	m := gpt2.NewModel("gpt2/models/base/model.onnx", "0", gpt2.NewDefaultConfig())

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	if out, err := m.Generate(prompt, 5, nil); err != nil {
		log.Fatal(err)
	} else {
		fmt.Printf("\n%v\n", out)
	}

	if err := m.Destroy(); err != nil {
		log.Fatal(err)
	}
}
