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
	static []string
}

func NewFSA() *FSA {
	return &FSA{
		static: []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"},
	}
}

func (f *FSA) Read(state int, next rune) (int, bool) {
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

	if state == StateInitial {
		switch r {
		case RuneUnicode32:
			return StateU32, true
		case RuneWhitespaceNotUnicode32:
			return StateWhitespaceNotUnicode32, true
		case RuneLetter:
			return StateLetter, true
		case RuneNumber:
			return StateNumber, true
		default:
			return StateOther, true
		}
	} else if state == StateU32 {
		switch r {
		case RuneUnicode32:
			return StateWhitespaceLookAhead, true
		case RuneWhitespaceNotUnicode32:
			return StateWhitespaceLookAhead, true
		case RuneLetter:
			return StateLetter, true
		case RuneNumber:
			return StateNumber, true
		default:
			return StateOther, true
		}
	} else if state == StateWhitespaceNotUnicode32 {
		switch r {
		case RuneUnicode32:
			return StateWhitespaceLookAhead, true
		case RuneWhitespaceNotUnicode32:
			return StateWhitespaceLookAhead, true
		default:
			return state, false
		}
	} else if state == StateNumber {
		switch r {
		case RuneNumber:
			return StateNumber, true
		default:
			return state, false
		}
	} else if state == StateLetter {
		switch r {
		case RuneLetter:
			return StateLetter, true
		default:
			return state, false
		}
	} else if state == StateOther {
		switch r {
		case RuneOther:
			return StateOther, true
		default:
			return state, false
		}
	} else if state == StateWhitespaceLookAhead {
		switch r {
		case RuneUnicode32:
			return StateWhitespaceLookAhead, true
		case RuneWhitespaceNotUnicode32:
			return StateWhitespaceLookAhead, true
		default:
			return state, false
		}
	} else {
		panic("invalid state")
	}
}

func (f *FSA) FindAll(s string) []string {
	if !utf8.ValidString(s) {
		// s = strings.ToValidUTF8(s, string(utf8.RuneError)) replaces with a single utf8.RuneError
		// s = strings.Map(func(r rune) rune { return r }, s) should be identical to s = string([]rune(s))

		s = string([]rune(s)) // replaces each invalid byte with utf8.RuneError during decoding
	}

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

		state := StateInitial

		for stop < len(s) {
			r, size := utf8.DecodeRuneInString(s[stop:])

			next, ok := f.Read(state, r)

			if ok {
				prev = stop
				stop += size

				state = next
			} else {
				if state == StateInitial {
					return matches
				}

				if state == StateWhitespaceLookAhead {
					matches = append(matches, s[start:prev])

					return findAll(prev, matches)
				}

				matches = append(matches, s[start:stop])

				return findAll(stop, matches)
			}
		}

		if start < len(s) {
			matches = append(matches, s[start:])
		}

		return matches
	}

	return findAll(0, make([]string, 0))
}
