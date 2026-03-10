package llm

type result struct {
	value  float64
	n      int
	tokens []int
	logits []float32
	doc    int
}
