package tui

import (
	"fmt"
	"strings"
	"time"
)

type ProgressBar struct {
	title string
	width int
	steps int
	n     int
	start time.Time
}

func NewProgressBar(title string, width, steps int, start time.Time) *ProgressBar {
	pb := &ProgressBar{
		title: title,
		width: width,
		steps: steps,
		start: start,
	}

	pb.Print()

	return pb
}

func (pb *ProgressBar) Increment() {
	pb.n++
}

func (pb *ProgressBar) Update(steps int) {
	pb.n = steps
}

func (pb *ProgressBar) Print() {
	fmt.Print(pb.String())
}

func (pb *ProgressBar) Finish() {
	fmt.Println()
}

func (pb *ProgressBar) String() string {
	progress := int(float64(pb.n) / float64(pb.steps) * float64(pb.width))
	percentage := int(float64(pb.n) / float64(pb.steps) * 100)

	lap := time.Since(pb.start)

	duration := fmt.Sprintf("%02d:%02d:%02d", int(lap.Hours()), int(lap.Minutes())%60, int(lap.Seconds())%60)

	return fmt.Sprintf(
		"\r[%s] %-20s %3d%% [%s%s] %d/%d",
		duration,
		pb.title,
		percentage,
		strings.Repeat("=", min(progress, pb.width)),
		strings.Repeat("-", max(0, pb.width-progress)),
		pb.n,
		pb.steps,
	)
}
