package nn_test

import (
	"github.com/DeedleFake/nn"
	"math"
	"math/rand"
	"testing"
)

func TestAllOutputs(t *testing.T) {
	n := nn.New(3, 5, 2)
	t.Log(n.AllOutputs([]float64{3, 2, 1}))
}

func TestTrain(t *testing.T) {
	const (
		Alpha = 1
	)

	n := nn.New(1, 10, 1)
	for i := 0; i < 10000; i++ {
		r := rand.Float64() * 10
		n.Train(
			[]float64{r},
			[]float64{math.Sin(r)},
			Alpha,
		)
	}

	for i := -.5; i <= .5; i += .2 {
		t.Logf("%v -> (%v, %v)\n", i, n.Run([]float64{i}), math.Sin(i))
	}
}
