package split

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	RuneUnicode32 = iota
	RuneWhitespaceNotUnicode32
	RuneLetter
	RuneNumber
	RuneOther
	StateInitial
	StateU32
	StateWhitespaceNotUnicode32
	StateLetter
	StateNumber
	StateOther
	StateWhitespaceLookAhead
)

type FSA struct {
	state  int
	static []string
}

func NewFSA() *FSA {
	return &FSA{
		state:  StateInitial,
		static: []string{"'s", "'t", "'re", "'m", "'ll", "'d"},
	}
}

func (f *FSA) Reset() {
	f.state = StateInitial
}

func (f *FSA) Read(next rune) bool {
	var r int

	if next == 32 {
		r = RuneUnicode32
	} else if unicode.IsSpace(next) {
		r = RuneWhitespaceNotUnicode32
	} else if unicode.IsLetter(next) {
		r = RuneLetter
	} else if unicode.IsNumber(next) {
		r = RuneNumber
	} else {
		r = RuneOther
	}

	if f.state == StateInitial {
		switch r {
		case RuneUnicode32:
			f.state = StateU32

			break
		case RuneWhitespaceNotUnicode32:
			f.state = StateWhitespaceNotUnicode32

			break
		case RuneLetter:
			f.state = StateLetter

			break
		case RuneNumber:
			f.state = StateNumber

			break
		default:
			f.state = StateOther
		}
	} else if f.state == StateU32 {
		switch r {
		case RuneUnicode32:
			f.state = StateWhitespaceLookAhead

			break
		case RuneWhitespaceNotUnicode32:
			f.state = StateWhitespaceLookAhead

			break
		case RuneLetter:
			f.state = StateLetter

			break
		case RuneNumber:
			f.state = StateNumber

			break
		default:
			f.state = StateOther
		}
	} else if f.state == StateWhitespaceNotUnicode32 {
		switch r {
		case RuneUnicode32:
			f.state = StateWhitespaceLookAhead

			break
		case RuneWhitespaceNotUnicode32:
			f.state = StateWhitespaceLookAhead

			break
		default:
			return false
		}
	} else if f.state == StateNumber {
		switch r {
		case RuneNumber:
			f.state = StateNumber

			break
		default:
			return false
		}
	} else if f.state == StateLetter {
		switch r {
		case RuneLetter:
			f.state = StateLetter

			break
		default:
			return false
		}
	} else if f.state == StateOther {
		switch r {
		case RuneOther:
			f.state = StateOther

			break
		default:
			return false
		}
	} else if f.state == StateWhitespaceLookAhead {
		switch r {
		case RuneUnicode32:
			f.state = StateWhitespaceLookAhead

			break
		case RuneWhitespaceNotUnicode32:
			f.state = StateWhitespaceLookAhead

			break
		default:
			return false
		}
	} else {
		panic("invalid state")
	}

	return true
}

func (f *FSA) FindAll(s string) []string {
	var findAll func(start int, matches []string) []string

	findAll = func(start int, matches []string) []string {
		if start >= len(s) {
			return matches
		}

		for _, v := range f.static {
			if strings.HasPrefix(s[start:], v) {
				matches = append(matches, v)

				next := start + len(v)

				if next >= len(s) {
					return matches
				}

				return findAll(next, matches)
			}
		}

		prev, stop := start, start

		for stop < len(s) {
			r, size := utf8.DecodeRuneInString(s[stop:])

			ok := f.Read(r)

			if ok {
				prev = stop
				stop += size
			} else {
				if f.state == StateInitial {
					return matches
				}

				if f.state == StateWhitespaceLookAhead {
					matches = append(matches, s[start:prev])

					f.Reset()

					return findAll(prev, matches)
				}

				matches = append(matches, s[start:stop])

				f.Reset()

				return findAll(stop, matches)
			}
		}

		if start < len(s) {
			matches = append(matches, s[start:])
		}

		return matches
	}

	defer f.Reset()

	return findAll(0, make([]string, 0))
}
