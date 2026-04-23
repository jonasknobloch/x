package shelf

import (
	"bufio"
	"bytes"
	"path/filepath"
	"strings"

	"github.com/go-git/go-git/v5/plumbing/format/gitignore"
)

type Ignore struct {
	matcher gitignore.Matcher
}

func newIgnore(data []byte) *Ignore {
	var patterns []gitignore.Pattern

	scanner := bufio.NewScanner(bytes.NewReader(data))

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		patterns = append(patterns, gitignore.ParsePattern(line, nil))
	}

	return &Ignore{
		matcher: gitignore.NewMatcher(patterns),
	}
}

func (i *Ignore) Match(path string, isDir bool) bool {
	parts := strings.Split(filepath.ToSlash(path), "/")

	var segments []string

	for _, p := range parts {
		if p != "" {
			segments = append(segments, p)
		}
	}

	return i.matcher.Match(segments, isDir)
}
