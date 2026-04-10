package lesci

import (
	"context"
	"database/sql"
	"image/color"
	"path"
	"slices"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func (e *Experiment) Plot(db *sql.DB) error {
	var rows *sql.Rows

	if r, err := db.QueryContext(context.Background(), "SELECT tok, mean, treat FROM lesci_results"); err != nil {
		return err
	} else {
		rows = r
	}

	defer rows.Close()

	var pts plotter.XYs

	var inX, inY, oovX, oovY []float64

	for rows.Next() {
		var tok, mean float64

		var treat bool

		if err := rows.Scan(&tok, &mean, &treat); err != nil {
			return err
		}

		if mean < -20 || mean > 0 {
			continue
		}

		pts = append(pts, plotter.XY{X: tok, Y: mean})

		if treat {
			inX = append(inX, tok)
			inY = append(inY, mean)
		} else {
			oovX = append(oovX, tok)
			oovY = append(oovY, mean)
		}
	}

	if err := rows.Err(); err != nil {
		return err
	}

	return render(pts, inX, inY, oovX, oovY, float64(e.cutoff), path.Join(e.name, "scatter.png"))
}

func render(pts plotter.XYs, inX, inY, oovX, oovY []float64, cutoff float64, out string) error {
	p := plot.New()

	p.Title.Text = "Original Data and Regression Fit for Mean"

	p.X.Label.Text = "Token"
	p.Y.Label.Text = "Mean"

	p.Y.Min = -20
	p.Y.Max = 0

	p.Add(plotter.NewGrid())

	if err := addScatter(p, pts); err != nil {
		return err
	}

	orangeRed := color.RGBA{R: 255, G: 69, B: 0, A: 255}

	aIn, bIn := stat.LinearRegression(inX, inY, nil, false)
	aOov, bOov := stat.LinearRegression(oovX, oovY, nil, false)

	addLine(p, aIn, bIn, slices.Min(inX), slices.Max(inX), orangeRed)
	addLine(p, aOov, bOov, cutoff, slices.Max(oovX), orangeRed)

	if err := addStep(p, aIn, bIn, aOov, bOov, cutoff, orangeRed); err != nil {
		return err
	}

	return p.Save(10*vg.Inch, 6*vg.Inch, out)
}

func addScatter(p *plot.Plot, pts plotter.XYs) error {
	s, err := plotter.NewScatter(pts)

	if err != nil {
		return err
	}

	s.Color = color.RGBA{A: 64}
	s.Radius = vg.Points(2)
	s.Shape = draw.CircleGlyph{}

	p.Add(s)

	return nil
}

func addLine(p *plot.Plot, alpha, beta, xMin, xMax float64, c color.Color) {
	f := plotter.NewFunction(func(x float64) float64 {
		return alpha + beta*x
	})

	f.Color = c
	f.Width = vg.Points(1.5)
	f.XMin = xMin
	f.XMax = xMax

	p.Add(f)
}

func addStep(p *plot.Plot, aIn, bIn, aOov, bOov, cutoff float64, c color.Color) error {
	l, err := plotter.NewLine(plotter.XYs{
		{X: cutoff, Y: aIn + bIn*cutoff},
		{X: cutoff, Y: aOov + bOov*cutoff},
	})

	if err != nil {
		return err
	}

	l.Color = c
	l.Width = vg.Points(1.5)

	p.Add(l)

	return nil
}
