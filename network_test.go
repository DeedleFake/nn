package nn_test

import (
	"github.com/DeedleFake/nn"
	"math"
	"testing"
)

const (
	Alpha = .1
)

func assertSliceEqual(t *testing.T, s1, s2 []float64) (r bool) {
	defer func() {
		if !r {
			t.Errorf("Assertion failed:\n\tExpected: %v\n\tGot: %v", s2, s1)
		}
	}()

	if len(s1) != len(s2) {
		return false
	}

	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}

	return true
}

func TestAllOutputs(t *testing.T) {
	n := nn.New(3, 5, 2)
	t.Log(n.AllOutputs([]float64{3, 2, 1}))
}

func TestPanics(t *testing.T) {
	nn.New(1, 3, 5).Train([]float64{0}, []float64{0, 0, 0, 0, 0}, .1)
	nn.New(5, 3, 1).Train([]float64{0, 0, 0, 0, 0}, []float64{0}, .1)
}

func TestNeuron(t *testing.T) {
	n := nn.Neuron{1, 2, 3}
	if got := n.Input([]float64{2, 3}); got != 14 {
		t.Errorf("%v != %v", got, 14)
	}

	n.Update([]float64{2, 3}, 10, 3)
	assertSliceEqual(t, n, []float64{31, 62, 93})
}

func TestXOR(t *testing.T) {
	n := nn.New(2, 2, 1)
	for i := 0; i < 100000; i++ {
		_, err := n.Train(
			[]float64{0, 0},
			[]float64{0},
			Alpha,
		)

		_, tmp := n.Train(
			[]float64{0, 1},
			[]float64{1},
			Alpha,
		)
		err += tmp

		_, tmp = n.Train(
			[]float64{1, 0},
			[]float64{1},
			Alpha,
		)
		err += tmp

		_, err = n.Train(
			[]float64{1, 1},
			[]float64{0},
			Alpha,
		)
		err += tmp

		if i%10000 == 0 {
			t.Logf("Averge error after %v iterations: %v", i, err/4)
		}
	}
	t.Logf("%v ^ %v -> %v", 0, 0, n.Run([]float64{0, 0}))
	t.Logf("%v ^ %v -> %v", 0, 1, n.Run([]float64{0, 1}))
	t.Logf("%v ^ %v -> %v", 1, 0, n.Run([]float64{1, 0}))
	t.Logf("%v ^ %v -> %v", 1, 1, n.Run([]float64{1, 1}))
}

func TestDiv(t *testing.T) {
	n := nn.New(1, 1)
	for i := 0; i < 100000; i++ {
		for i := 0; i < 10; i++ {
			n.Train(
				[]float64{float64(i)},
				[]float64{float64(i) / 2},
				Alpha,
			)
		}
	}

	for i := float64(0); i < 10; i++ {
		t.Logf("%v / 2 -> %v", i, n.Run([]float64{i}))
	}
}

func TestSin(t *testing.T) {
	n := nn.New(1, 10, 10, 1)
	for i := 0; i < 100000; i++ {
		for in := -math.Pi; in <= math.Pi; in += .05 {
			n.Train(
				[]float64{in},
				[]float64{math.Sin(in)},
				Alpha,
			)
		}
	}

	for i := -.5; i <= .5; i += .2 {
		t.Logf("sin(%v) -> (%v, %v)\n", i, n.Run([]float64{i}), math.Sin(i))
	}
}
