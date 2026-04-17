package sander

import (
	"context"
	"database/sql"
	"image/color"
	"path"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func (e *Experiment) Plot(db *sql.DB) error {
	var rows *sql.Rows

	if r, err := db.QueryContext(context.Background(), `
		SELECT s.token_id, s.distance, e.reference
		FROM similarity_buffer s
		JOIN embeddings e ON s.token_id = e.token_id
		ORDER BY s.token_id ASC
	`); err != nil {
		return err
	} else {
		rows = r
	}

	defer rows.Close()

	var pts plotter.XYZs

	for rows.Next() {
		var id, distance float64
		var reference bool

		if err := rows.Scan(&id, &distance, &reference); err != nil {
			return err
		}

		z := 0.0

		if reference {
			z = 1.0
		}

		pts = append(pts, plotter.XYZ{X: id, Y: distance, Z: z})
	}

	if err := rows.Err(); err != nil {
		return err
	}

	return render(pts, path.Join(e.name, "similarity_scatter.png"))
}

func render(pts plotter.XYZs, out string) error {
	p := plot.New()

	p.Y.Label.Text = "Euclidean Distance"

	p.Y.Min = 0
	p.Y.Max = 5

	var candidate, reference plotter.XYZs

	for _, pt := range pts {
		if pt.Z == 1.0 {
			reference = append(reference, pt)
		} else {
			candidate = append(candidate, pt)
		}
	}

	if err := addScatter(p, candidate, color.RGBA{A: 64}); err != nil {
		return err
	}

	if err := addScatter(p, reference, color.RGBA{R: 255, G: 69, B: 0, A: 255}); err != nil {
		return err
	}

	return p.Save(10*vg.Inch, 6*vg.Inch, out)
}

func addScatter(p *plot.Plot, pts plotter.XYZs, c color.Color) error {
	s, err := plotter.NewScatter(pts)

	if err != nil {
		return err
	}

	s.Color = c
	s.Radius = vg.Points(2)
	s.Shape = draw.CircleGlyph{}

	p.Add(s)

	return nil
}
