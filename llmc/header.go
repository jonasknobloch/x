package llmc

import "encoding/binary"

const headerWords = 256

type Header struct {
	Magic   int32
	Version int32
	NumToks int32
}

func (h Header) encode() [headerWords * 4]byte {
	var buffer [headerWords * 4]byte

	binary.LittleEndian.PutUint32(buffer[0:], uint32(h.Magic))
	binary.LittleEndian.PutUint32(buffer[4:], uint32(h.Version))
	binary.LittleEndian.PutUint32(buffer[8:], uint32(h.NumToks))

	return buffer
}
