package shelf

import "path"

type Item string

func Abs(item Item) string {
	return path.Join(Root, string(item))
}
