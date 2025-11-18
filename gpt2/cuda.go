package gpt2

import ort "github.com/yalue/onnxruntime_go"

func SessionsOptionsWithCUDADeviceID(deviceID string) (*ort.SessionOptions, error) {
	var sessionOptions *ort.SessionOptions
	var cudaProviderOptions *ort.CUDAProviderOptions

	if s, err := ort.NewSessionOptions(); err != nil {
		return nil, err
	} else {
		sessionOptions = s
	}

	if c, err := ort.NewCUDAProviderOptions(); err != nil {
		return nil, err
	} else {
		cudaProviderOptions = c
	}

	if err := cudaProviderOptions.Update(map[string]string{
		"device_id": deviceID,
	}); err != nil {
		return nil, err
	}

	if err := sessionOptions.AppendExecutionProviderCUDA(cudaProviderOptions); err != nil {
		return nil, err
	}

	if err := cudaProviderOptions.Destroy(); err != nil {
		return nil, err
	}

	return sessionOptions, nil
}
