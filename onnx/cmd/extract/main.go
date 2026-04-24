package main

import (
	"fmt"
	"log"

	"go.jknobloc.com/x/onnx"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tensor"
)

func main() {
	data, shape, err := onnx.ExtractInitializer(shelf.Abs("models/gpt2/model.onnx"), "transformer.wte.weight")

	if err != nil {
		log.Fatal(err)
	}

	t := tensor.NewDense[float32](shape, data)

	eot := t.Select(0, 50256).Contiguous()

	fmt.Println(eot)
}
