package sander

import (
	"go.jknobloc.com/x/tokenizer/bpe"
)

func UnusedTokensGPT2(tokenizer *bpe.Tokenizer) map[int64]struct{} {
	unused := map[int64]struct{}{
		177: {},
		178: {},
		179: {},
		180: {},
		181: {},
		182: {},
		183: {},
		184: {},
		185: {},
		186: {},
		187: {},
	}

	alphabet := bpe.InitialAlphabet()

	itoa := bpe.Itoa(tokenizer)

	for id := range unused {
		token, ok := itoa[id]

		if !ok {
			panic("unknown token id")
		}

		if token != string(alphabet[id]) {
			panic("unexpected token")
		}
	}

	return unused
}

func UnusedTokensMBPE(tokenizer *bpe.Tokenizer) map[int64]struct{} {
	unused := make(map[int64]struct{})

	vocab := bpe.Vocab(tokenizer)

	mask := bpe.ReachableTokens(tokenizer, vocab)

	for i := range vocab {
		if mask[i] {
			continue
		}

		unused[int64(i)] = struct{}{}
	}

	return unused
}

// TODO we could just filter some input data
