package bpe

type Config struct {
	Recover bool
}

func DefaultConfig() Config {
	return Config{
		Recover: false,
	}
}
