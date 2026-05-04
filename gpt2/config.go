package gpt2

type Config struct {
	VocabSize    int
	NumLayers    int
	NumHeads     int
	HeadDim      int
	NumPositions int
}

type Options struct {
	WithCache    bool
	WithLogits   bool
	WithLogProbs bool
}

func ConfigDefault() Config {
	return Config{
		VocabSize:    50257,
		NumLayers:    12,
		NumHeads:     12,
		HeadDim:      64,
		NumPositions: 1024,
	}
}

func ConfigLarge() Config {
	return Config{
		VocabSize:    50257,
		NumLayers:    36,
		NumHeads:     20,
		HeadDim:      64,
		NumPositions: 1024,
	}
}
