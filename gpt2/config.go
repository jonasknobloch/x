package gpt2

type Config struct {
	vocabSize  int
	nLayers    int
	nHeads     int
	headDim    int
	nPositions int
}

func DefaultConfig() Config {
	return Config{
		vocabSize:  50257,
		nLayers:    12,
		nHeads:     12,
		headDim:    64,
		nPositions: 1024,
	}
}

func (c Config) WithVocabSize(vocabSize int) Config {
	c.vocabSize = vocabSize

	return c
}
