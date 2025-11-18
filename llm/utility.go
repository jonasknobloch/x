package llm

func toInt64(s []int) []int64 {
	r := make([]int64, len(s))

	for i, v := range s {
		r[i] = int64(v)
	}

	return r
}

func toInt(s []int64) []int {
	r := make([]int, len(s))

	for i, v := range s {
		r[i] = int(v)
	}

	return r
}
