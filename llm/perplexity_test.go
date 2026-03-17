package llm

import (
	"fmt"
	"testing"
)

func TestEvaluator_estimateJobs(t *testing.T) {
	type gold struct {
		tokens   int
		window   int
		stride   int
		expected int
	}

	tests := []gold{
		{tokens: 20, window: 10, stride: 3, expected: 4},
		{tokens: 20, window: 10, stride: 4, expected: 3},
		{tokens: 20, window: 10, stride: 5, expected: 3},

		{tokens: 0, window: 1, stride: 1, expected: 0},
		{tokens: 1, window: 1, stride: 1, expected: 1},

		{tokens: 0, window: 1024, stride: 1, expected: 0},
		{tokens: 1, window: 1, stride: 1024, expected: 1},
	}

	e := NewEvaluator[any]()

	for _, tt := range tests {
		t.Run(
			fmt.Sprintf("tokens%d_window%d_stride%d", tt.tokens, tt.window, tt.stride),

			func(t *testing.T) {
				tokens := make([]int64, tt.tokens)

				got := e.estimateJobs(tokens, tt.window, tt.stride)

				if got != tt.expected {
					t.Errorf("expected %d but got (%d, %d, %d) = %d", tt.expected, tt.tokens, tt.window, tt.stride, got)
				}
			},
		)
	}
}
