package shelf

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
)

func hashFile(name string) (string, error) {
	var file *os.File

	if f, err := os.Open(name); err != nil {
		return "", err
	} else {
		file = f
	}

	defer file.Close()

	hash := sha256.New()

	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}

	return hex.EncodeToString(hash.Sum(nil)), nil
}
