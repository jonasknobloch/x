package gpt2

import (
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	batchSize = 1
	seqLen    = 128
)

func BenchmarkStandard(b *testing.B) {
	shape := ort.NewShape(batchSize, seqLen)
	data := make([]int64, batchSize*seqLen)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		t, err := ort.NewTensor(shape, data)

		if err != nil {
			b.Fatal(err)
		}

		t.Destroy()
	}
}

func BenchmarkPersistent(b *testing.B) {
	shape := ort.NewShape(batchSize, seqLen)
	data := make([]int64, batchSize*seqLen)

	persistent, err := ort.NewEmptyTensor[int64](shape)

	if err != nil {
		b.Fatal(err)
	}

	defer persistent.Destroy()

	buffer := persistent.GetData()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		copy(buffer, data)
	}
}
