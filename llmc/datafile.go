package llmc

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"golang.org/x/exp/constraints"
)

type ModelDesc string

const (
	GPT2   ModelDesc = "gpt-2"
	LLaMA3 ModelDesc = "llama-3"
)

type modelInfo struct {
	magic   int32
	version int32
	wide    bool // true: uint32 tokens; false: uint16 tokens
}

var headerInfo = map[ModelDesc]modelInfo{
	GPT2:   {magic: 20240520, version: 1, wide: false},
	LLaMA3: {magic: 20240801, version: 7, wide: true},
}

type DataFile[T constraints.Integer] struct {
	Model  ModelDesc
	Tokens []T // token IDs
}

func Deserialize[T constraints.Integer](name string, dst *DataFile[T]) (int, error) {
	var file *os.File

	if f, err := os.Open(name); err != nil {
		return 0, err
	} else {
		file = f
	}

	defer file.Close()

	var header [headerWords * 4]byte

	if _, err := io.ReadFull(file, header[:]); err != nil {
		return 0, err
	}

	magic := int32(binary.LittleEndian.Uint32(header[0:]))
	version := int32(binary.LittleEndian.Uint32(header[4:]))
	numToks := int32(binary.LittleEndian.Uint32(header[8:]))

	var model ModelDesc
	var info modelInfo

	for m, mi := range headerInfo {
		if mi.magic == magic && mi.version == version {
			model, info = m, mi
			break
		}
	}

	if model == "" {
		return 0, fmt.Errorf("unknown magic:version %d:%d", magic, version)
	}

	tokens := make([]T, numToks)

	if info.wide {
		tokensU32 := make([]uint32, numToks)

		if err := binary.Read(file, binary.LittleEndian, tokensU32); err != nil {
			return 0, err
		}

		for i, t := range tokensU32 {
			tokens[i] = T(t)
		}
	} else {
		tokensU16 := make([]uint16, numToks)

		if err := binary.Read(file, binary.LittleEndian, tokensU16); err != nil {
			return 0, err
		}

		for i, t := range tokensU16 {
			tokens[i] = T(t)
		}
	}

	dst.Model = model
	dst.Tokens = tokens

	return int(numToks), nil
}

func Serialize[T constraints.Integer](src *DataFile[T], name string) (int, error) {
	info, ok := headerInfo[src.Model]

	if !ok {
		return 0, fmt.Errorf("unknown model descriptor %q", src.Model)
	}

	h := Header{
		Magic:   info.magic,
		Version: info.version,
		NumToks: int32(len(src.Tokens)),
	}

	var file *os.File

	if f, err := os.Create(name); err != nil {
		return 0, err
	} else {
		file = f
	}

	defer file.Close()

	header := h.encode()

	if _, err := file.Write(header[:]); err != nil {
		return 0, err
	}

	if info.wide {
		tokensU32 := make([]uint32, len(src.Tokens))

		for i, t := range src.Tokens {
			tokensU32[i] = uint32(t)
		}

		if err := binary.Write(file, binary.LittleEndian, tokensU32); err != nil {
			return 0, err
		}
	} else {
		tokensU16 := make([]uint16, len(src.Tokens))

		for i, t := range src.Tokens {
			tokensU16[i] = uint16(t)
		}

		if err := binary.Write(file, binary.LittleEndian, tokensU16); err != nil {
			return 0, err
		}
	}

	return len(src.Tokens), nil
}
