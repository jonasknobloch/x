package lesci

type Config struct {
	ClampRulesBeforeFilter bool
}

type Options struct {
	ForceContext bool
	ForceExtract bool
}

func ConfigDefault() Config {
	return Config{
		ClampRulesBeforeFilter: true,
	}
}
