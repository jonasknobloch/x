package onnx

//go:generate bash scripts/gen.sh

import (
	"encoding/binary"
	"math"
	"os"

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
			raw := init.GetRawData()

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
