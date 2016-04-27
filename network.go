package nn

import (
	"math/rand"
)

// A Network is a neural network.
type Network struct {
	// G is the activator function for the network.
	G func(float64) float64

	// GPrime is the derivative of the activator function. It is the
	// responisiblity of the client to ensure that the value of GPrime
	// cooresponds with the value of G.
	GPrime func(float64) float64

	// w is the actual network. The first index into w is the layer,
	// starting with the first hidden layer and ending with the output
	// layer. The second index is the neuron in that layer.
	w [][]Neuron
}

// New creates a new neural network with the given number of inputs
// and the given number of neurons per layer.
//
// The values of layers will coorespond to the number of neurons per
// hidden layer as well as the output layer. For example,
//
//     New(2, 2, 1)
//
// will produce a network with the following structure:
//
//     I───H
//      ╲ ╱ ╲
//       ╳   O
//      ╱ ╲ ╱
//     I───H
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

// Run returns the output of the network with the given inputs.
func (nn *Network) Run(data []float64) (out []float64) {
	// TODO: Bounds checking?

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

// AllOutputs is similar to Run, but returns the outputs of every
// layer, not just the output layer. Index 0 of out is the input
// slice, and the last index is the same as what Run would have
// returned.
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

// Train trains the network with a single example. It returns the
// outputs of the layers, similar to AllOutputs, and the accuracy of
// the final output relative to the expected output.
//
// BUG: This doesn't work for some reason.
func (nn *Network) Train(in, out []float64, alpha float64) ([][]float64, float64) {
	// This function is based on the following psuedocode:
	//
	//     inputs:
	//       x:       Input to the network.
	//       y:       Expected output of the network.
	//       L:       Number of layers in the network.
	//       w[i][j]: Weight of the connection from i to j.
	//       g:       Activator function.
	//       alpha:   Learning rate.
	//
	//     for each node i in the input layer do
	//       a[i] = x[i]
	//     for lay = 2 to L do
	//       for each node j in layer lay do
	//         in[j] = sum over all i in w[i][j] * a[i]
	//         a[j] = g(in[j])
	//
	//     for each node j in the output layer do
	//       delta[j] = g'(in[j]) * (y[j] - a[j])
	//     for lay = L - 1 to 1 do
	//       for each node i in layer lay do
	//         delta[i] = g'(in[j]) * (sum over all j of w[i][j] * delta[j])
	//
	//     for each weight w[i][j] do
	//       w[i][j] += alpha * a[i] * delta[j]

	a := nn.AllOutputs(in)
	delta := make([][]float64, len(nn.w))
	outLayer := len(nn.w) - 1

	var acc float64
	for neuron := range nn.w[outLayer] {
		gp := nn.GPrime(nn.w[outLayer][neuron].Input(a[outLayer]))
		e := out[neuron] - a[outLayer+1][neuron]
		delta[outLayer] = append(delta[outLayer], gp*e)

		acc += (e * e) / 2
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

	return a, acc / float64(len(out))
}

// A Neuron is a single node in the network, represented as a list of
// weights cooresponding to its own inputs, with the exception of
// index 0, which is for the weight of a bias input and does not
// coorespond to an actual input to the neuron.
type Neuron []float64

// Input returns a weighted sum of the input slice, including the
// bias input. len(in) should equal len(n) - 1. The output of this
// method should be the input to the activator function.
func (n Neuron) Input(in []float64) float64 {
	sum := n[0]
	for i := range in {
		sum += n[i+1] * in[i]
	}

	return sum
}

// Update updates the weights of the neuron.
func (n Neuron) Update(in []float64, alpha, delta float64) {
	n[0] += alpha * delta
	for i := 1; i < len(n); i++ {
		n[i] += alpha * in[i-1] * delta
	}
}
