package gpt2

type Config struct {
	VocabSize    int
	NumLayers    int
	NumHeads     int
	HeadDim      int
	NumPositions int
}

func DefaultConfig() Config {
	return Config{
		VocabSize:    50257,
		NumLayers:    12,
		NumHeads:     12,
		HeadDim:      64,
		NumPositions: 1024,
	}
}
