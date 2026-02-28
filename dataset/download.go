package dataset

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
)

const Base = "https://datasets-server.huggingface.co/parquet"

func Download(dataset, config, root string) error {
	var pl *parquetListing

	if l, err := listing(dataset, config); err != nil {
		return err
	} else {
		pl = l
	}

	splits := pl.Splits()

	for _, split := range splits {
		dir := filepath.Join(root, split)

		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}

		total := int64(0)

		for pf := range pl.Split(split) {
			name := filepath.Join(dir, filepath.Base(pf.Filename))

			if n, err := download(pf, name); err != nil {
				return err
			} else {
				total += n

				fmt.Printf("[%s] downloaded %s (%d bytes), total=%d\n", split, filepath.Base(name), n, total)
			}
		}
	}

	return nil
}

type parquetListing struct {
	ParquetFiles []parquetFile `json:"parquet_files"`
}

type parquetFile struct {
	Dataset  string `json:"dataset"`
	Config   string `json:"config"`
	Split    string `json:"split"`
	URL      string `json:"url"`
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
}

func (pl *parquetListing) Splits() []string {
	seen := make(map[string]struct{})

	for _, ps := range pl.ParquetFiles {
		if ps.Split == "" {
			continue // TODO possible?
		}

		seen[ps.Split] = struct{}{}
	}

	splits := make([]string, 0)

	for k := range seen {
		splits = append(splits, k)
	}

	slices.Sort(splits)

	return splits
}

func (pl *parquetListing) Split(split string) iter.Seq[parquetFile] {
	return func(yield func(pf parquetFile) bool) {
		for _, pf := range pl.ParquetFiles {
			if pf.Split != split {
				continue
			}

			if !yield(pf) {
				return
			}
		}
	}
}

func listing(dataset, config string) (*parquetListing, error) {
	var parquet *url.URL

	if u, err := url.Parse(Base); err != nil {
		return nil, err
	} else {
		parquet = u
	}

	query := parquet.Query()

	query.Set("dataset", dataset)
	query.Set("config", config)

	parquet.RawQuery = query.Encode()

	var req *http.Request

	if r, err := http.NewRequest("GET", parquet.String(), nil); err != nil {
		return nil, err
	} else {
		req = r
	}

	var resp *http.Response

	if r, err := http.DefaultClient.Do(req); err != nil {
		return nil, err
	} else {
		resp = r

		defer resp.Body.Close()
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("http not ok")
	}

	var pl parquetListing

	if err := json.NewDecoder(resp.Body).Decode(&pl); err != nil {
		return nil, err
	}

	return &pl, nil
}

func download(pf parquetFile, name string) (int64, error) {
	if st, err := os.Stat(name); err == nil && st.Size() == pf.Size {
		return pf.Size, nil
	}

	var req *http.Request

	if r, err := http.NewRequest("GET", pf.URL, nil); err != nil {
		return 0, err
	} else {
		req = r
	}

	var resp *http.Response

	if r, err := http.DefaultClient.Do(req); err != nil {
		return 0, err
	} else {
		resp = r

		defer resp.Body.Close()
	}

	if resp.StatusCode != http.StatusOK {
		return 0, errors.New("http not ok")
	}

	tmp := name + ".part"

	var file *os.File

	ok := false

	if f, err := os.Create(tmp); err != nil {
		return 0, err
	} else {
		file = f
	}

	defer func() {
		_ = file.Close()

		if !ok {
			_ = os.Remove(tmp)
		}
	}()

	written := int64(0)

	if n, err := io.Copy(file, resp.Body); err != nil {
		return 0, err
	} else {
		written = n
	}

	if err := file.Sync(); err != nil {
		return 0, err
	}

	if pf.Size > 0 && written != pf.Size {
		return 0, errors.New("size mismatch")
	}

	if err := file.Close(); err != nil {
		return 0, err
	}

	if err := os.Rename(tmp, name); err != nil {
		return 0, err
	}

	ok = true

	return written, nil
}
