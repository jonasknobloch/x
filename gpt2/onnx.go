package gpt2

import ort "github.com/yalue/onnxruntime_go"

func InitializeEnvironment() error {
	ort.SetSharedLibraryPath(SharedLibraryPath())

	return ort.InitializeEnvironment()
}

func DestroyEnvironment() error {
	return ort.DestroyEnvironment()
}
