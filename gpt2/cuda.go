package gpt2

import ort "github.com/yalue/onnxruntime_go"

func WithCUDAProvider(options *ort.SessionOptions, deviceID string) error {
	var cudaProviderOptions *ort.CUDAProviderOptions

	if c, err := ort.NewCUDAProviderOptions(); err != nil {
		return err
	} else {
		cudaProviderOptions = c
	}

	if err := cudaProviderOptions.Update(map[string]string{
		"device_id": deviceID,
	}); err != nil {
		return err
	}

	if err := options.AppendExecutionProviderCUDA(cudaProviderOptions); err != nil {
		return err
	}

	if err := cudaProviderOptions.Destroy(); err != nil {
		return err
	}

	return nil
}
