package gpt2

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type Allocator struct {
	config         Config
	options        Options
	batchSize      int
	sequenceLength int64
	step           int64
	inputNames     []string
	outputNames    []string
	values         map[string]ort.Value
}

func NewAllocator(cfg Config, opts Options, batchSize int) *Allocator {
	return &Allocator{
		config:       cfg,
		options:      opts,
		batchSize:    batchSize,
		values:       make(map[string]ort.Value),
	}
}

func (a *Allocator) InputNames() []string {
	capacity := 3

	if a.options.WithCache {
		capacity += 2 * a.config.NumLayers
	}

	names := make([]string, 0, capacity)

	names = append(names, "input_ids", "position_ids", "attention_mask")

	if a.options.WithCache {
		for i := range a.config.NumLayers {
			names = append(names, fmt.Sprintf("past_key_values.%d.key", i), fmt.Sprintf("past_key_values.%d.value", i))
		}
	}

	return names
}

func (a *Allocator) OutputNames() []string {
	capacity := 0

	if a.options.WithLogits {
		capacity++
	}

	if a.options.WithLogProbs {
		capacity++
	}

	if a.options.WithCache {
		capacity += 2 * a.config.NumLayers
	}

	names := make([]string, 0, capacity)

	if a.options.WithLogits {
		names = append(names, "logits")
	}

	if a.options.WithLogProbs {
		names = append(names, "token_logprobs")
	}

	if a.options.WithCache {
		for i := range a.config.NumLayers {
			names = append(names, fmt.Sprintf("present.%d.key", i), fmt.Sprintf("present.%d.value", i))
		}
	}

	return names
}

func (a *Allocator) SetBatchSize(n int) {
	a.batchSize = n
}

func (a *Allocator) Init(tokens []int64) error {
	if len(tokens)%a.batchSize != 0 {
		panic("token count is not divisible by batch size")
	}

	a.Destroy()

	a.values = make(map[string]ort.Value)
	a.sequenceLength = int64(len(tokens) / a.batchSize)
	a.step = 0

	if err := a.initInputs(tokens); err != nil {
		a.Destroy()

		return err
	}

	if err := a.initOutputs(tokens); err != nil {
		a.Destroy()

		return err
	}

	a.step = a.sequenceLength

	return nil
}

func (a *Allocator) initInputs(tokens []int64) error {
	capacity := 3

	if a.options.WithCache {
		capacity += 2 * a.config.NumLayers
	}

	names := make([]string, 0, capacity)

	if err := a.inputIDs(tokens, false); err != nil {
		return err
	}

	names = append(names, "input_ids")

	if err := a.positionIDs(0, false); err != nil {
		return err
	}

	names = append(names, "position_ids")

	if err := a.attentionMask(0, false); err != nil {
		return err
	}

	names = append(names, "attention_mask")

	if !a.options.WithCache {
		a.inputNames = names

		return nil
	}

	for i := range int64(a.config.NumLayers) {
		if err := a.pastKeyValues(i, "key", false); err != nil {
			return err
		}

		names = append(names, fmt.Sprintf("past_key_values.%d.key", i))

		if err := a.pastKeyValues(i, "value", false); err != nil {
			return err
		}

		names = append(names, fmt.Sprintf("past_key_values.%d.value", i))
	}

	a.inputNames = names

	return nil
}

func (a *Allocator) initOutputs(tokens []int64) error {
	capacity := 0

	if a.options.WithLogits {
		capacity++
	}

	if a.options.WithLogProbs {
		capacity++
	}

	if a.options.WithCache {
		capacity += 2 * a.config.NumLayers
	}

	names := make([]string, 0, capacity)

	if a.options.WithLogits {
		if err := a.logits(false); err != nil {
			return err
		}

		names = append(names, "logits")
	}

	if a.options.WithLogProbs {
		if err := a.logProbs(false); err != nil {
			return err
		}

		names = append(names, "token_logprobs")
	}

	if !a.options.WithCache {
		a.outputNames = names

		return nil
	}

	for i := range int64(a.config.NumLayers) {
		if err := a.presentKeyValues(0, i, "key", false); err != nil {
			return err
		}

		names = append(names, fmt.Sprintf("present.%d.key", i))

		if err := a.presentKeyValues(0, i, "value", false); err != nil {
			return err
		}

		names = append(names, fmt.Sprintf("present.%d.value", i))
	}

	a.outputNames = names

	return nil
}

func (a *Allocator) Step(token int64) error {
	if a.batchSize != 1 {
		panic("step requires batch size 1")
	}

	tokens := []int64{token}

	a.sequenceLength = 1

	if err := a.inputIDs(tokens, true); err != nil {
		return err
	}

	if err := a.positionIDs(a.step, true); err != nil {
		return err
	}

	if err := a.attentionMask(a.step, true); err != nil {
		return err
	}

	if a.options.WithLogits {
		if err := a.logits(true); err != nil {
			return err
		}
	}

	if a.options.WithLogProbs {
		if err := a.logProbs(true); err != nil {
			return err
		}
	}

	for i := range int64(a.config.NumLayers) {
		for _, suffix := range []string{"key", "value"} {
			if err := a.rotateCache(i, suffix); err != nil {
				return err
			}
		}
	}

	a.step++

	return nil
}

func (a *Allocator) Destroy() {
	for _, v := range a.values {
		_ = v.Destroy()
	}
}

func (a *Allocator) Inputs() ([]string, []ort.Value) {
	vals := make([]ort.Value, len(a.inputNames))

	for i, n := range a.inputNames {
		vals[i] = a.values[n]
	}

	return a.inputNames, vals
}

func (a *Allocator) Outputs() ([]string, []ort.Value) {
	vals := make([]ort.Value, len(a.outputNames))

	for i, n := range a.outputNames {
		vals[i] = a.values[n]
	}

	return a.outputNames, vals
}

func (a *Allocator) Value(name string) ort.Value {
	v, ok := a.values[name]

	if !ok {
		panic("unknown value: " + name)
	}

	return v
}

func (a *Allocator) inputIDs(tokens []int64, force bool) error {
	const name = "input_ids"

	if _, ok := a.values[name]; ok {
		if !force {
			panic("input_ids already allocated")
		}

		_ = a.values[name].Destroy()
	}

	shape := []int64{int64(a.batchSize), a.sequenceLength}

	if t, err := ort.NewTensor[int64](shape, tokens); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) positionIDs(start int64, force bool) error {
	const name = "position_ids"

	if _, ok := a.values[name]; ok {
		if !force {
			panic("position_ids already allocated")
		}

		_ = a.values[name].Destroy()
	}

	data := make([]int64, int64(a.batchSize)*a.sequenceLength)
	shape := []int64{int64(a.batchSize), a.sequenceLength}

	for b := range int64(a.batchSize) {
		for s := range a.sequenceLength {
			data[b*a.sequenceLength+s] = start + s
		}
	}

	if t, err := ort.NewTensor[int64](shape, data); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) attentionMask(start int64, force bool) error {
	const name = "attention_mask"

	if _, ok := a.values[name]; ok {
		if !force {
			panic("attention_mask already allocated")
		}

		_ = a.values[name].Destroy()
	}

	data := make([]int64, int64(a.batchSize)*(start+a.sequenceLength))
	shape := []int64{int64(a.batchSize), start + a.sequenceLength}

	for i := range data {
		data[i] = 1
	}

	if t, err := ort.NewTensor[int64](shape, data); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) pastKeyValues(i int64, suffix string, force bool) error {
	if int(i) > a.config.NumLayers {
		panic("invalid layer index")
	}

	if suffix != "key" && suffix != "value" {
		panic("invalid suffix")
	}

	name := fmt.Sprintf("past_key_values.%d.%s", i, suffix)

	if _, ok := a.values[name]; ok {
		if !force {
			panic(name + " already allocated")
		}

		_ = a.values[name].Destroy()
	}

	shape := []int64{int64(a.batchSize), int64(a.config.NumHeads), 0, int64(a.config.HeadDim)}

	if t, err := ort.NewEmptyTensor[float32](shape); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) logits(force bool) error {
	const name = "logits"

	if _, ok := a.values[name]; ok {
		if !force {
			panic("logits already allocated")
		}

		_ = a.values[name].Destroy()
	}

	shape := []int64{int64(a.batchSize), a.sequenceLength, int64(a.config.VocabSize)}

	if t, err := ort.NewEmptyTensor[float32](shape); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) logProbs(force bool) error {
	const name = "token_logprobs"

	if _, ok := a.values[name]; ok {
		if !force {
			panic("token_logprobs already allocated")
		}

		_ = a.values[name].Destroy()
	}

	shape := []int64{int64(a.batchSize), a.sequenceLength - 1}

	if t, err := ort.NewEmptyTensor[float32](shape); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) presentKeyValues(start, i int64, suffix string, force bool) error {
	if int(i) > a.config.NumLayers {
		panic("invalid layer index")
	}

	if suffix != "key" && suffix != "value" {
		panic("invalid suffix")
	}

	name := fmt.Sprintf("present.%d.%s", i, suffix)

	if _, ok := a.values[name]; ok {
		if !force {
			panic(name + " already allocated")
		}

		_ = a.values[name].Destroy()
	}

	shape := []int64{int64(a.batchSize), int64(a.config.NumHeads), start + a.sequenceLength, int64(a.config.HeadDim)}

	if t, err := ort.NewEmptyTensor[float32](shape); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) rotateCache(i int64, suffix string) error {
	past := fmt.Sprintf("past_key_values.%d.%s", i, suffix)
	present := fmt.Sprintf("present.%d.%s", i, suffix)

	if v, ok := a.values[past]; ok {
		_ = v.Destroy()
	}

	a.values[past] = a.values[present]

	delete(a.values, present)

	if err := a.presentKeyValues(a.step, i, suffix, false); err != nil {
		return err
	}

	return nil
}
