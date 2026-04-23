package shelf

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"io/fs"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"
)

var Root = ".shelf"

type Sum map[Item]string

func Index(ignore []byte) (Sum, error) {
	sum := make(Sum)

	var rw sync.RWMutex

	g, ctx := errgroup.WithContext(context.Background())

	sem := make(chan struct{}, max(1, runtime.NumCPU()-1))

	ign := newIgnore(ignore)

	rootPath := filepath.Clean(Root)

	walkErr := filepath.WalkDir(rootPath, func(abs string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}

		rel, relErr := filepath.Rel(rootPath, abs)

		if relErr != nil {
			return err
		}

		if rel == "." {
			return nil
		}

		if ign.Match(rel, d.IsDir()) {
			if d.IsDir() {
				return fs.SkipDir
			}

			return nil
		}

		if d.IsDir() || !d.Type().IsRegular() {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case sem <- struct{}{}:
		}

		g.Go(func() error {
			defer func() { <-sem }()

			hash, hashErr := hashFile(abs)

			if hashErr != nil {
				return hashErr
			}

			rw.Lock()

			sum[Item(rel)] = hash

			rw.Unlock()

			return nil
		})

		return nil
	})

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return sum, walkErr
}

func Serialize(s Sum, w io.Writer) error {
	items := make([]Item, 0, len(s))

	for item := range s {
		items = append(items, item)
	}

	slices.SortFunc(items, func(a, b Item) int {
		return strings.Compare(string(a), string(b))
	})

	for _, item := range items {
		if _, err := fmt.Fprintf(w, "%s  %s\n", s[item], item); err != nil {
			return err
		}
	}

	return nil
}

func Deserialize(r io.Reader) (Sum, error) {
	sum := make(Sum)

	s := bufio.NewScanner(r)

	for s.Scan() {
		line := s.Text()

		parts := strings.SplitN(line, "  ", 2)

		if len(parts) != 2 {
			return nil, fmt.Errorf("malformed line: %q", line)
		}

		sum[Item(parts[1])] = parts[0]
	}

	return sum, s.Err()
}
