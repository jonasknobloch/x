package gpt2

import (
	"errors"
	"fmt"
	_ "llm"
	"math"
	"os"
	"slices"
	"sort"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	vocabSize  = 50257
	nLayers    = 12
	nHeads     = 12
	headDim    = 64
	nPositions = 1024
)

type Model struct {
	name        string
	deviceID    string
	session     *ort.DynamicAdvancedSession
	inputNames  []string
	outputNames []string
}

func NewModel(name, deviceID string) *Model {
	return &Model{
		name:     name,
		deviceID: deviceID,
	}
}

func SharedLibraryPath() string {
	p, ok := os.LookupEnv("ONNXRUNTIME_SHARED_LIBRARY_PATH")

	if !ok {
		// TODO embed runtime binaries
	}

	return p
}

func IntraOpNumThreads() int {
	s, ok := os.LookupEnv("ONNXRUNTIME_INTRA_OP_NUM_THREADS")

	if !ok {
		return 0
	}

	n, err := strconv.Atoi(s)

	if err != nil {
		return 0
	}

	return n
}

func (m *Model) Init() error {
	ort.SetSharedLibraryPath(SharedLibraryPath())

	if err := ort.InitializeEnvironment(); err != nil {
		return err
	}

	inputNames := make([]string, 0, 3+2*nLayers)
	outputNames := make([]string, 0, 1+2*nLayers)

	inputNames = append(inputNames, "input_ids", "position_ids", "attention_mask")
	outputNames = append(outputNames, "logits")

	for i := range nLayers {
		inputNames = append(inputNames, fmt.Sprintf("past_key_values.%d.key", i), fmt.Sprintf("past_key_values.%d.value", i))
		outputNames = append(outputNames, fmt.Sprintf("present.%d.key", i), fmt.Sprintf("present.%d.value", i))
	}

	m.inputNames = inputNames
	m.outputNames = outputNames

	var options *ort.SessionOptions

	if o, err := ort.NewSessionOptions(); err != nil {
		return err
	} else {
		options = o

		defer options.Destroy()
	}

	if m.deviceID != "" {
		if err := WithCUDAProvider(options, m.deviceID); err != nil {
			return err
		}
	}

	if n := IntraOpNumThreads(); n > 0 {
		if err := options.SetIntraOpNumThreads(n); err != nil {
			return err
		}
	}

	if s, err := ort.NewDynamicAdvancedSession(m.name, m.inputNames, m.outputNames, options); err != nil {
		return err
	} else {
		m.session = s
	}

	return nil
}

func (m *Model) Destroy() error {
	return ort.DestroyEnvironment()
}

func (m *Model) Generate(prompt []int64, steps int64, logits *[][]float32) ([]int64, error) {
	if len(prompt) == 0 {
		return nil, errors.New("empty prompt")
	}

	if int64(len(prompt))+steps > nPositions {
		return nil, errors.New("sequence length exceeds context limit")
	}

	n := int64(len(prompt))
	r := make([]int64, steps)

	outputs := initCache()

	if o, err := m.forward(prompt, 0, outputs); err != nil {
		defer destroyValues(outputs)

		return nil, err
	} else {
		destroyValues(outputs)

		outputs = o
	}

	for step := range steps {
		l := m.logits(outputs[0])

		if logits != nil {
			*logits = append(*logits, l...)
		}

		next := m.sample(l[len(l)-1])

		r[step] = next

		_ = outputs[0].Destroy()

		if o, err := m.forward([]int64{next}, n+step, outputs[1:]); err != nil {
			defer destroyValues(outputs[1:])

			return nil, err
		} else {
			destroyValues(outputs[1:])

			outputs = o
		}
	}

	if logits != nil {
		for _, l := range m.logits(outputs[0]) {
			*logits = append(*logits, l)
		}
	}

	_ = outputs[0].Destroy()

	return r, nil
}

func (m *Model) logits(output ort.Value) [][]float32 {
	d := output.(*ort.Tensor[float32]).GetData()
	n := len(d) / vocabSize
	l := make([][]float32, n)

	for i := range n {
		s := i * vocabSize

		l[i] = d[s : s+vocabSize : s+vocabSize]
	}

	return l
}

func (m *Model) sample(logits []float32) int64 {
	idx, _ := topK(softmax(logits), 5)

	return int64(idx[0])
}

func (m *Model) forward(tokens []int64, start int64, cache []ort.Value) ([]ort.Value, error) {
	var binding *ort.IoBinding

	if b, err := m.session.CreateIoBinding(); err != nil {
		return nil, err
	} else {
		binding = b

		defer binding.Destroy()
	}

	var inputs []ort.Value
	var outputs []ort.Value

	if in, err := initInputs(tokens, start); err != nil {
		return nil, err
	} else {
		inputs = in
	}

	defer destroyValues(inputs)

	if out, err := initOutputs(tokens, start); err != nil {
		return nil, err
	} else {
		outputs = out
	}

	inputs = append(inputs, cache...)

	if len(inputs) != len(m.inputNames) {
		panic("unexpected input length")
	}

	for i, name := range m.inputNames {
		if err := binding.BindInput(name, inputs[i]); err != nil {
			return nil, err
		}
	}

	var ok bool

	defer func() {
		if !ok {
			destroyValues(outputs)
		}
	}()

	if len(outputs) != len(m.outputNames) {
		panic("unexpected output length")
	}

	for i, name := range m.outputNames {
		if err := binding.BindOutput(name, outputs[i]); err != nil {
			return nil, err
		}
	}

	if err := m.session.RunWithBinding(binding); err != nil {
		return nil, err
	}

	ok = true

	return outputs, nil
}

func destroyValues(values []ort.Value) {
	for _, v := range values {
		_ = v.Destroy()
	}
}

func initCache() []ort.Value {
	values := make([]ort.Value, 0, 2*nLayers)
	shape := []int64{1, int64(nHeads), 0, int64(headDim)}

	for range nLayers {
		kTensor, _ := ort.NewEmptyTensor[float32](shape)
		vTensor, _ := ort.NewEmptyTensor[float32](shape)

		values = append(values, ort.Value(kTensor), ort.Value(vTensor))
	}

	return values
}

func initInputs(tokens []int64, start int64) ([]ort.Value, error) {
	var inputIDs *ort.Tensor[int64]
	var positionIDs *ort.Tensor[int64]
	var attentionMask *ort.Tensor[int64]

	inputsShape := []int64{1, int64(len(tokens))}
	inputsData := tokens

	if t, err := ort.NewTensor[int64](inputsShape, inputsData); err != nil {
		return nil, err
	} else {
		inputIDs = t
	}

	positionsShape := []int64{1, int64(len(tokens))}
	positionsData := make([]int64, len(tokens))

	for i := range len(tokens) {
		positionsData[i] = start + int64(i)
	}

	if p, err := ort.NewTensor[int64](positionsShape, positionsData); err != nil {
		return nil, err
	} else {
		positionIDs = p
	}

	maskData := make([]int64, start+int64(len(tokens)))
	maskShape := []int64{1, start + int64(len(tokens))}

	for i := range maskData {
		maskData[i] = 1
	}

	if m, err := ort.NewTensor[int64](maskShape, maskData); err != nil {
		return nil, err
	} else {
		attentionMask = m
	}

	inputs := []ort.Value{ort.Value(inputIDs), ort.Value(positionIDs), ort.Value(attentionMask)}

	return inputs, nil
}

func initOutputs(tokens []int64, start int64) ([]ort.Value, error) {
	outputs := make([]ort.Value, 0, 1+2*nLayers)

	logits, err := ort.NewEmptyTensor[float32]([]int64{1, int64(len(tokens)), int64(vocabSize)})

	if err != nil {
		return nil, err
	}

	outputs = append(outputs, ort.Value(logits))

	shape := []int64{1, int64(nHeads), start + int64(len(tokens)), int64(headDim)}

	for range nLayers {
		kTensor, _ := ort.NewEmptyTensor[float32](shape)
		vTensor, _ := ort.NewEmptyTensor[float32](shape)

		outputs = append(outputs, ort.Value(kTensor), ort.Value(vTensor))
	}

	return outputs, nil
}

func softmax(logits []float32) []float32 {
	m := slices.Max(logits)

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
