package main

import (
	"fmt"
	"log"
	"math"
	"sort"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	nLayers = 12
	nHeads  = 12
	headDim = 64
)

func main() {
	ort.SetSharedLibraryPath("lib/onnxruntime-osx-arm64-1.22.0/lib/libonnxruntime.1.22.0.dylib")

	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}

	defer ort.DestroyEnvironment()

	inputNames := []string{"input_ids", "position_ids", "attention_mask"}
	outputNames := []string{"logits"}

	tokens, _ := ort.NewTensor[int64]([]int64{1, 1}, []int64{464})
	positions, _ := ort.NewTensor[int64]([]int64{1, 1}, []int64{0})
	attentionMask, _ := ort.NewTensor[int64]([]int64{1, 1}, []int64{1})
	logits, _ := ort.NewEmptyTensor[float32]([]int64{1, 1, 50257})

	inputs := []ort.Value{ort.Value(tokens), ort.Value(positions), ort.Value(attentionMask)}
	outputs := []ort.Value{ort.Value(logits)}

	cacheNames, cacheValues := emptyCache()

	inputNames = append(inputNames, cacheNames...)
	inputs = append(inputs, cacheValues...)

	session, err := ort.NewAdvancedSession(
		"scripts/onnx-gpt2/model.onnx",
		inputNames,
		outputNames,
		inputs,
		outputs,
		nil,
	)

	if err != nil {
		log.Fatal(err)
	}

	defer session.Destroy()

	if err := session.Run(); err != nil {
		log.Fatal(err)
	}

	data := logits.GetData()
	probs := softmax(data)

	idx, p := topK(probs, 10)

	for i, t := range idx {
		fmt.Printf("%.4f [%d]\n", p[i], t)
	}
}

func emptyCache() ([]string, []ort.Value) {
	names := make([]string, 0, 2*nLayers)
	values := make([]ort.Value, 0, 2*nLayers)
	shape := []int64{1, int64(nHeads), 0, int64(headDim)}

	for i := range nLayers {
		kName := fmt.Sprintf("past_key_values.%d.key", i)
		vName := fmt.Sprintf("past_key_values.%d.value", i)

		kTensor, _ := ort.NewEmptyTensor[float32](shape)
		vTensor, _ := ort.NewEmptyTensor[float32](shape)

		names = append(names, kName, vName)
		values = append(values, ort.Value(kTensor), ort.Value(vTensor))
	}

	return names, values
}

func softmax(logits []float32) []float32 {
	m := logits[0]

	for _, v := range logits {
		if v > m {
			m = v
		}
	}

	s := float32(0.0)
	r := make([]float32, len(logits))

	for i, v := range logits {
		e := float32(math.Exp(float64(v - m)))

		r[i] = e
		s += e
	}

	for i := range r {
		r[i] /= s
	}

	return r
}

func topK(p []float32, k int) ([]int, []float32) {
	n := len(p)

	if k > n {
		k = n
	}

	idx := make([]int, n)

	for i := range idx {
		idx[i] = i
	}

	sort.Slice(idx, func(i, j int) bool {
		return p[idx[i]] > p[idx[j]]
	})

	topIdx := idx[:k]
	topP := make([]float32, k)

	for i := 0; i < k; i++ {
		topP[i] = p[topIdx[i]]
	}

	return topIdx, topP
}
