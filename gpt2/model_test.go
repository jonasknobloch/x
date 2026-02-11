package gpt2

import (
	"encoding/binary"
	"log"
	"math"
	"os"
	"slices"
	"testing"
)

func fromModel() []float32 {
	prompt := []int64{464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290}

	m := model()

	defer m.Destroy()

	logits := make([][]float32, 0)

	if _, err := m.Generate(prompt, 0, &logits); err != nil {
		log.Fatal(err)
	}

	return flatten(logits)
}

func model() *Model {
	m := NewModel("models/base/model.onnx", "0") // TODO check if CUDA is available

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	return m
}

func fromGold() []float32 {
	shape := []int{1, 9, 50257}

	s, err := f32("test/logits.f32", shape[0]*shape[1]*shape[2])

	if err != nil {
		log.Fatal(err)
	}

	return s
}

func f32(name string, n int) ([]float32, error) {
	var file *os.File

	if f, err := os.Open(name); err != nil {
		return nil, err
	} else {
		file = f
	}

	data := make([]float32, n)

	if err := binary.Read(file, binary.LittleEndian, data); err != nil {
		return nil, err
	}

	return data, nil
}

func TestModel_GenerateTopK(t *testing.T) {
	a := fromGold()
	b := fromModel()

	shape := []int{1, 9, 50257}

	for i := range shape[0] {
		for j := range shape[1] {
			start := (i * shape[1] * shape[2]) + (j * shape[2])
			stop := start + shape[2]

			topA, _ := topK(a[start:stop], 64)
			topB, _ := topK(b[start:stop], 64)

			if !slices.Equal(topA, topB) {
				t.Errorf("token mismatch at token %d batch %d", j, i)
			}
		}
	}
}

func TestModel_GenerateRMSE(t *testing.T) {
	a := fromGold()
	b := fromModel()

	e := rmse(a, b)

	if e > 0 {
		t.Fatal("RMSE exceeds threshold")
	}
}

func flatten(s [][]float32) []float32 {
	n := 0

	for _, row := range s {
		n += len(row)
	}

	r := make([]float32, n)

	i := 0

	for _, row := range s {
		for _, v := range row {
			r[i] = v
			i++
		}
	}

	return r
}

func mse(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("length mismatch")
	}

	s := float32(0)

	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}

	return s / float32(len(a))
}

func rmse(a, b []float32) float32 {
	return float32(math.Sqrt(float64(mse(a, b))))
}
