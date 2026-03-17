package tui

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"time"
)

type ProgressBar struct {
	title     string
	width     int
	total     atomic.Int64
	completed atomic.Int64
	start     time.Time
}

func NewProgressBar(title string, width, total int, start time.Time) *ProgressBar {
	pb := &ProgressBar{
		title: title,
		width: width,
		start: start,
	}

	pb.total.Store(int64(total))

	pb.Print()

	return pb
}

func (pb *ProgressBar) SetTotal(steps int) {
	pb.total.Store(int64(steps))
}

func (pb *ProgressBar) Total() int {
	return int(pb.total.Load())
}

func (pb *ProgressBar) SetCompleted(steps int) {
	pb.completed.Store(int64(steps))
}

func (pb *ProgressBar) Completed() int {
	return int(pb.completed.Load())
}

func (pb *ProgressBar) SetProgress(completed, total int) {
	pb.completed.Store(int64(completed))
	pb.total.Store(int64(total))
}

func (pb *ProgressBar) Progress() (int, int) {
	return pb.Completed(), pb.Total()
}

func (pb *ProgressBar) Add(steps int) {
	pb.completed.Add(int64(steps))
}

func (pb *ProgressBar) Print() {
	fmt.Print(pb.String())
}

func (pb *ProgressBar) Finish() {
	fmt.Println(pb.String())
}

func (pb *ProgressBar) String() string {
	title := pb.title
	width := pb.width
	total := pb.Total()
	completed := pb.Completed()
	start := pb.start

	var progress int
	var percentage int

	if total > 0 {
		progress = int(float64(completed) / float64(total) * float64(width))
		percentage = int(float64(completed) / float64(total) * 100)
	}

	lap := time.Since(start)

	duration := fmt.Sprintf("%02d:%02d:%02d", int(lap.Hours()), int(lap.Minutes())%60, int(lap.Seconds())%60)

	return fmt.Sprintf(
		"\r[%s] %-20s %3d%% [%s%s] %d/%d",
		duration,
		title,
		percentage,
		strings.Repeat("=", min(progress, width)),
		strings.Repeat("-", max(0, width-progress)),
		completed,
		total,
	)
}

func (pb *ProgressBar) Watch(
	ctx context.Context,
	interval time.Duration,
	callback func() int,
) {
	ticker := time.NewTicker(interval)

	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			n := callback()

			pb.SetCompleted(n)

			pb.Finish()

			return

		case <-ticker.C:
			n := callback()

			pb.SetCompleted(n)

			pb.Print()
		}
	}
}
