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

func (p *ProgressBar) Increment() {
	p.n++
}

func (p *ProgressBar) Update(steps int) {
	p.n = steps
}

func (p *ProgressBar) Print() {
	fmt.Print(p.String())
}

func (p *ProgressBar) Finish() {
	fmt.Println()
}

func (p *ProgressBar) String() string {
	progress := int(float64(p.n) / float64(p.steps) * float64(p.width))
	percentage := int(float64(p.n) / float64(p.steps) * 100)

	lap := time.Since(p.start)

	duration := fmt.Sprintf("%02d:%02d:%02d", int(lap.Hours()), int(lap.Minutes())%60, int(lap.Seconds())%60)

	return fmt.Sprintf(
		"\r[%s] %-20s %3d%% [%s%s] %d/%d",
		duration,
		p.title,
		percentage,
		strings.Repeat("=", min(progress, p.width)),
		strings.Repeat("-", max(0, p.width-progress)),
		p.n,
		p.steps,
	)
}
