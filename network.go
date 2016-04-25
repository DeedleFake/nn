package nn

import (
	"fmt"
	"math/rand"
)

type Network struct {
	G      func(float64) float64
	GPrime func(float64) float64

	w [][][]float64
}

func New(inputs int, layers ...int) *Network {
	w := make([][][]float64, len(layers))
	for layer := range w {
		w[layer] = make([][]float64, layers[layer])
		for neuron := range w[layer] {
			prev := inputs
			if layer > 0 {
				prev = layers[layer-1]
			}

			w[layer][neuron] = make([]float64, prev)
			for input := range w[layer][neuron] {
				w[layer][neuron][input] = rand.Float64()
			}
		}
	}

	return &Network{
		G:      Sigmoid,
		GPrime: SigmoidPrime,

		w: w,
	}
}

func (nn *Network) Run(data ...float64) (out [][]float64) {
	if len(data) != len(nn.w[0][0]) {
		panic(fmt.Errorf("Expected %v inputs, but got %v", len(nn.w[0][0]), len(data)))
	}

	out = make([][]float64, 1, len(nn.w)+1)
	out[0] = data

	for layer := range nn.w {
		cur := make([]float64, 0, len(nn.w[layer]))
		for neuron := range nn.w[layer] {
			var sum float64
			for input := range nn.w[layer][neuron] {
				sum += nn.w[layer][neuron][input] * out[layer][input]
			}

			cur = append(cur, nn.G(sum))
		}

		out = append(out, cur)
	}

	return out
}
