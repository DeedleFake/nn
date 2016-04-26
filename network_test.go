package nn_test

import (
	"github.com/DeedleFake/nn"
	"math"
	"testing"
)

func TestAllOutputs(t *testing.T) {
	n := nn.New(3, 5, 2)
	t.Log(n.AllOutputs([]float64{3, 2, 1}))
}

func TestTrain(t *testing.T) {
	const (
		Alpha = .5
	)

	n := nn.New(1, 10, 10, 1)
	for i := 0; i < 10000; i++ {
		for in := -math.Pi; in <= math.Pi; in += .05 {
			n.Train(
				[]float64{in},
				[]float64{math.Sin(in)},
				Alpha,
			)
		}
	}

	for i := -.5; i <= .5; i += .2 {
		t.Logf("%v -> (%v, %v)\n", i, n.Run([]float64{i}), math.Sin(i))
	}
}
