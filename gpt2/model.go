package gpt2

import (
	"errors"
	"math"
	"os"
	"slices"
	"sort"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"
)

type Model struct {
	name         string
	deviceID     string
	config       Config
	withCache    bool
	withLogits   bool
	withLogProbs bool
	session      *ort.DynamicAdvancedSession
	allocator    *Allocator
}

func NewModel(name string, deviceID string, config Config, withCache bool, withLogits bool, withLogProbs bool) *Model {
	return &Model{
		name:         name,
		deviceID:     deviceID,
		config:       config,
		withCache:    withCache,
		withLogits:   withLogits,
		withLogProbs: withLogProbs,
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
	m.allocator = NewAllocator(m.config, 1, m.withCache, m.withLogits, m.withLogProbs)

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

	if s, err := ort.NewDynamicAdvancedSession(m.name, m.allocator.InputNames(), m.allocator.OutputNames(), options); err != nil {
		return err
	} else {
		m.session = s
	}

	return nil
}

func (m *Model) Destroy() {
	m.allocator.Destroy()
}

func (m *Model) Generate(prompt []int64, steps int64, logits *[][]float32) ([]int64, error) {
	if !m.withLogits {
		panic("generate requires logits output")
	}

	if steps > 0 && !m.withCache {
		panic("generate with steps > 0 requires cache")
	}

	if len(prompt) == 0 {
		return nil, errors.New("empty prompt")
	}

	if int64(len(prompt))+steps > int64(m.config.nPositions) {
		return nil, errors.New("sequence length exceeds context limit")
	}

	if err := m.allocator.Init(prompt); err != nil {
		return nil, err
	}

	if err := m.forward(m.allocator); err != nil {
		return nil, err
	}

	r := make([]int64, steps)

	for step := range steps {
		l := m.logits(m.allocator.Value("logits"))

		if logits != nil {
			*logits = append(*logits, l...)
		}

		next := m.sample(l[len(l)-1])

		r[step] = next

		if err := m.allocator.Step(next); err != nil {
			return nil, err
		}

		if err := m.forward(m.allocator); err != nil {
			return nil, err
		}
	}

	if logits != nil {
		for _, l := range m.logits(m.allocator.Value("logits")) {
			*logits = append(*logits, l)
		}
	}

	return r, nil
}

func (m *Model) Score(tokens []int64, batchSize int, logProbs *[]float32) error {
	if !m.withLogProbs {
		panic("score requires token_logprobs output")
	}

	m.allocator.SetBatchSize(batchSize)

	if err := m.allocator.Init(tokens); err != nil {
		return err
	}

	if err := m.forward(m.allocator); err != nil {
		return err
	}

	if logProbs != nil {
		d := m.allocator.Value("token_logprobs").(*ort.Tensor[float32]).GetData()

		*logProbs = append(*logProbs, d...)
	}

	return nil
}

func (m *Model) logits(output ort.Value) [][]float32 {
	d := output.(*ort.Tensor[float32]).GetData()
	n := len(d) / m.config.vocabSize
	l := make([][]float32, n)

	for i := range n {
		s := i * m.config.vocabSize

		l[i] = d[s : s+m.config.vocabSize : s+m.config.vocabSize]
	}

	return l
}

func (m *Model) sample(logits []float32) int64 {
	idx, _ := topK(softmax(logits), 5)

	return int64(idx[0])
}

func (m *Model) forward(alloc *Allocator) error {
	var binding *ort.IoBinding

	if b, err := m.session.CreateIoBinding(); err != nil {
		return err
	} else {
		binding = b

		defer binding.Destroy()
	}

	inputNames, inputs := alloc.Inputs()
	outputNames, outputs := alloc.Outputs()

	for i, name := range inputNames {
		if err := binding.BindInput(name, inputs[i]); err != nil {
			return err
		}
	}

	for i, name := range outputNames {
		if err := binding.BindOutput(name, outputs[i]); err != nil {
			return err
		}
	}

	return m.session.RunWithBinding(binding)
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
