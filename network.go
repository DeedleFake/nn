package nn

import (
	"math/rand"
)

type Network struct {
	G      func(float64) float64
	GPrime func(float64) float64

	w [][]Neuron
}

func New(inputs int, layers ...int) *Network {
	w := make([][]Neuron, len(layers))
	for layer := range w {
		w[layer] = make([]Neuron, layers[layer])
		for neuron := range w[layer] {
			prev := inputs
			if layer > 0 {
				prev = layers[layer-1]
			}

			w[layer][neuron] = make(Neuron, prev+1)
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

func (nn *Network) Run(data []float64) (out []float64) {
	// TODO: This shares a lot of code with AllOutputs. Find a way to
	// reuse the code a bit better.
	out = make([]float64, len(data))
	copy(out, data)

	cur := make([]float64, 0, len(nn.w[0]))
	for layer := range nn.w {
		for neuron := range nn.w[layer] {
			cur = append(cur, nn.G(nn.w[layer][neuron].Input(out)))
		}

		out, cur = cur, out
		cur = cur[:0]
	}

	return out
}

func (nn *Network) AllOutputs(data []float64) (out [][]float64) {
	out = make([][]float64, 1, len(nn.w)+1)
	out[0] = data

	for layer := range nn.w {
		cur := make([]float64, 0, len(nn.w[layer]))
		for neuron := range nn.w[layer] {
			cur = append(cur, nn.G(nn.w[layer][neuron].Input(out[layer])))
		}

		out = append(out, cur)
	}

	return out
}

// BUG: This doesn't work for some reason.
func (nn *Network) Train(in, out []float64, alpha float64) [][]float64 {
	a := nn.AllOutputs(in)
	delta := make([][]float64, len(nn.w))
	outLayer := len(nn.w) - 1

	for neuron := range nn.w[outLayer] {
		gp := nn.GPrime(nn.w[outLayer][neuron].Input(a[outLayer]))
		e := out[neuron] - a[outLayer+1][neuron]
		delta[outLayer] = append(delta[outLayer], gp*e)
	}

	for layer := outLayer - 1; layer >= 0; layer-- {
		for neuron := range nn.w[layer] {
			var sum float64
			for next := range nn.w[layer+1] {
				sum += nn.w[layer+1][next][neuron+1] * delta[layer+1][next]
			}

			in := nn.w[layer][neuron].Input(a[layer])
			delta[layer] = append(delta[layer], nn.GPrime(in)*sum)
		}
	}

	for layer := range nn.w {
		for neuron := range nn.w[layer] {
			nn.w[layer][neuron].Update(a[layer], alpha, delta[layer][neuron])
		}
	}

	return a
}

type Neuron []float64

func (n Neuron) Input(in []float64) float64 {
	sum := n[0]
	for i := range in {
		sum += n[i+1] * in[i]
	}

	return sum
}

func (n Neuron) Update(in []float64, alpha, delta float64) {
	n[0] += alpha * delta
	for i := 1; i < len(n); i++ {
		n[i] += alpha * in[i-1] * delta
	}
}
