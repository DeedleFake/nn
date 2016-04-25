package nn

import (
	"math"
)

func Threshold(val float64) float64 {
	if val < 0 {
		return 0
	}

	return 1
}

func Sigmoid(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

func SigmoidPrime(val float64) float64 {
	return Sigmoid(val) * (1 - Sigmoid(val))
}
