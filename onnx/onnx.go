package onnx

//go:generate bash scripts/gen.sh

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"google.golang.org/protobuf/proto"

	"go.jknobloc.com/x/onnx/internal/pb"
)

func ExtractInitializer(name string, initializer string) ([]float32, []int, error) {
	var data []byte

	if d, err := os.ReadFile(name); err != nil {
		return nil, nil, err
	} else {
		data = d
	}

	model := &pb.ModelProto{}

	if err := proto.Unmarshal(data, model); err != nil {
		return nil, nil, err
	}

	g := model.GetGraph()

	var result []float32
	var shape []int64

	ok := false

	for _, init := range g.GetInitializer() {
		if init.GetName() == initializer {
			var raw []byte

			if init.GetDataLocation() == pb.TensorProto_EXTERNAL {
				raw = readExternalData(name, init.GetExternalData())
			} else {
				raw = init.GetRawData()
			}

			result = make([]float32, len(raw)/4)

			for i := 0; i < len(result); i++ {
				bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])

				result[i] = math.Float32frombits(bits)
			}

			shape = init.GetDims()

			ok = true

			break
		}
	}

	if !ok {
		panic("initializer not found")
	}

	shapeInt := make([]int, len(shape))

	for i, v := range shape {
		shapeInt[i] = int(v)
	}

	return result, shapeInt, nil
}

func readExternalData(name string, entries []*pb.StringStringEntryProto) []byte {
	var location string
	var offset, length int64

	for _, e := range entries {
		switch e.GetKey() {
		case "location":
			location = e.GetValue()
		case "offset":
			offset, _ = strconv.ParseInt(e.GetValue(), 10, 64)
		case "length":
			length, _ = strconv.ParseInt(e.GetValue(), 10, 64)
		}
	}

	data := filepath.Join(filepath.Dir(name), location)

	var file *os.File

	if f, err := os.Open(data); err != nil {
		panic(err)
	} else {
		file = f
	}

	defer file.Close()

	if offset > 0 {
		if _, err := file.Seek(offset, 0); err != nil {
			panic(err)
		}
	}

	var buf []byte

	if length > 0 {
		buf = make([]byte, length)

		if _, err := file.Read(buf); err != nil {
			panic(err)
		}
	} else {
		var err error

		buf, err = os.ReadFile(data)

		if err != nil {
			panic(err)
		}

		if offset > 0 {
			buf = buf[offset:]
		}
	}

	return buf
}
