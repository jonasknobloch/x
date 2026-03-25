package bpe

import (
	"github.com/jonasknobloch/mbpe"

	"go.jknobloc.com/x/tokenizer/bpe/split"
)

type ByteLevel struct {
	*mbpe.ByteLevel
	addPrefixSpace bool
	fsa            *split.FSA
}

func NewByteLevel(addPrefixSpace bool) *ByteLevel {
	return &ByteLevel{
		ByteLevel:      mbpe.NewByteLevel(addPrefixSpace),
		addPrefixSpace: addPrefixSpace,
		fsa:            split.NewFSA(),
	}
}

func (p *ByteLevel) PreTokenize(phrase string) []string {
	if phrase == "" {
		return []string{}
	}

	if p.addPrefixSpace && phrase[0] != ' ' {
		phrase = " " + phrase
	}

	compounds := p.fsa.FindAll(phrase)

	for i, compound := range compounds {
		r := ""

		for _, b := range []byte(compound) {
			r += mbpe.BytesChar[b]
		}

		compounds[i] = r
	}

	return compounds
}
