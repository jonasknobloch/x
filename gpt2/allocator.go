package gpt2

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type Allocator struct {
	config       Config
	step         int64
	inputNames   []string
	outputNames  []string
	values       map[string]ort.Value
	withCache    bool
	withLogits   bool
	withLogProbs bool
}

func NewAllocator(config Config, withCache bool, withLogits bool, withLogProbs bool) *Allocator {
	return &Allocator{
		config:       config,
		values:       make(map[string]ort.Value),
		withCache:    withCache,
		withLogits:   withLogits,
		withLogProbs: withLogProbs,
	}
}

func (a *Allocator) InputNames() []string {
	capacity := 3

	if a.withCache {
		capacity += 2 * a.config.nLayers
	}

	names := make([]string, 0, capacity)

	names = append(names, "input_ids", "position_ids", "attention_mask")

	if a.withCache {
		for i := range a.config.nLayers {
			names = append(names, fmt.Sprintf("past_key_values.%d.key", i), fmt.Sprintf("past_key_values.%d.value", i))
		}
	}

	return names
}

func (a *Allocator) OutputNames() []string {
	capacity := 0

	if a.withLogits {
		capacity++
	}

	if a.withLogProbs {
		capacity++
	}

	if a.withCache {
		capacity += 2 * a.config.nLayers
	}

	names := make([]string, 0, capacity)

	if a.withLogits {
		names = append(names, "logits")
	}

	if a.withLogProbs {
		names = append(names, "log_probs")
	}

	if a.withCache {
		for i := range a.config.nLayers {
			names = append(names, fmt.Sprintf("present.%d.key", i), fmt.Sprintf("present.%d.value", i))
		}
	}

	return names
}

func (a *Allocator) Init(tokens []int64) error {
	a.Destroy()

	a.values = make(map[string]ort.Value)
	a.step = 0

	if err := a.initInputs(tokens); err != nil {
		a.Destroy()

		return err
	}

	if err := a.initOutputs(tokens); err != nil {
		a.Destroy()

		return err
	}

	a.step = int64(len(tokens))

	return nil
}

func (a *Allocator) initInputs(tokens []int64) error {
	capacity := 3

	if a.withCache {
		capacity += 2 * a.config.nLayers
	}

	names := make([]string, 0, capacity)

	if err := a.inputIDs(tokens, false); err != nil {
		return err
	}

	names = append(names, "input_ids")

	if err := a.positionIDs(tokens, 0, false); err != nil {
		return err
	}

	names = append(names, "position_ids")

	if err := a.attentionMask(tokens, 0, false); err != nil {
		return err
	}

	names = append(names, "attention_mask")

	if !a.withCache {
		a.inputNames = names

		return nil
	}

	for i := range int64(a.config.nLayers) {
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

	if a.withLogits {
		capacity++
	}

	if a.withLogProbs {
		capacity++
	}

	if a.withCache {
		capacity += 2 * a.config.nLayers
	}

	names := make([]string, 0, capacity)

	if a.withLogits {
		if err := a.logits(tokens, false); err != nil {
			return err
		}

		names = append(names, "logits")
	}

	if a.withLogProbs {
		if err := a.logProbs(tokens, false); err != nil {
			return err
		}

		names = append(names, "log_probs")
	}

	if !a.withCache {
		a.outputNames = names

		return nil
	}

	for i := range int64(a.config.nLayers) {
		if err := a.presentKeyValues(tokens, 0, i, "key", false); err != nil {
			return err
		}

		names = append(names, fmt.Sprintf("present.%d.key", i))

		if err := a.presentKeyValues(tokens, 0, i, "value", false); err != nil {
			return err
		}

		names = append(names, fmt.Sprintf("present.%d.value", i))
	}

	a.outputNames = names

	return nil
}

func (a *Allocator) Step(token int64) error {
	tokens := []int64{token}

	if err := a.inputIDs(tokens, true); err != nil {
		return err
	}

	if err := a.positionIDs(tokens, a.step, true); err != nil {
		return err
	}

	if err := a.attentionMask(tokens, a.step, true); err != nil {
		return err
	}

	if a.withLogits {
		if err := a.logits(tokens, true); err != nil {
			return err
		}
	}

	if a.withLogProbs {
		if err := a.logProbs(tokens, true); err != nil {
			return err
		}
	}

	for i := range int64(a.config.nLayers) {
		for _, suffix := range []string{"key", "value"} {
			if err := a.rotateCache(tokens, i, suffix); err != nil {
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

	shape := []int64{1, int64(len(tokens))}

	if t, err := ort.NewTensor[int64](shape, tokens); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) positionIDs(tokens []int64, start int64, force bool) error {
	const name = "position_ids"

	if _, ok := a.values[name]; ok {
		if !force {
			panic("position_ids already allocated")
		}

		_ = a.values[name].Destroy()
	}

	data := make([]int64, len(tokens))
	shape := []int64{1, int64(len(tokens))}

	for i := range len(tokens) {
		data[i] = start + int64(i)
	}

	if t, err := ort.NewTensor[int64](shape, data); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) attentionMask(tokens []int64, start int64, force bool) error {
	const name = "attention_mask"

	if _, ok := a.values[name]; ok {
		if !force {
			panic("attention_mask already allocated")
		}

		_ = a.values[name].Destroy()
	}

	data := make([]int64, start+int64(len(tokens)))
	shape := []int64{1, start + int64(len(tokens))}

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
	if int(i) > a.config.nLayers {
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

	shape := []int64{1, int64(a.config.nHeads), 0, int64(a.config.headDim)}

	if t, err := ort.NewEmptyTensor[float32](shape); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) logits(tokens []int64, force bool) error {
	const name = "logits"

	if _, ok := a.values[name]; ok {
		if !force {
			panic("logits already allocated")
		}

		_ = a.values[name].Destroy()
	}

	shape := []int64{1, int64(len(tokens)), int64(a.config.vocabSize)}

	if t, err := ort.NewEmptyTensor[float32](shape); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) logProbs(tokens []int64, force bool) error {
	const name = "log_probs"

	if _, ok := a.values[name]; ok {
		if !force {
			panic("log_probs already allocated")
		}

		_ = a.values[name].Destroy()
	}

	shape := []int64{1, int64(len(tokens)) - 1}

	if t, err := ort.NewEmptyTensor[float32](shape); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) presentKeyValues(tokens []int64, start, i int64, suffix string, force bool) error {
	if int(i) > a.config.nLayers {
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

	shape := []int64{1, int64(a.config.nHeads), start + int64(len(tokens)), int64(a.config.headDim)}

	if t, err := ort.NewEmptyTensor[float32](shape); err != nil {
		return err
	} else {
		a.values[name] = ort.Value(t)
	}

	return nil
}

func (a *Allocator) rotateCache(tokens []int64, i int64, suffix string) error {
	past := fmt.Sprintf("past_key_values.%d.%s", i, suffix)
	present := fmt.Sprintf("present.%d.%s", i, suffix)

	if v, ok := a.values[past]; ok {
		_ = v.Destroy()
	}

	a.values[past] = a.values[present]

	delete(a.values, present)

	if err := a.presentKeyValues(tokens, a.step, i, suffix, false); err != nil {
		return err
	}

	return nil
}
