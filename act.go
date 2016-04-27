package nn

import (
	"math"
)

// Threshol implements a simple hard threshold function. If val is
// less than zero, it returns zero. Otherwise, it returns 1.
func Threshold(val float64) float64 {
	if val < 0 {
		return 0
	}

	return 1
}

// Sigmoid implements a sigmoid logistic function.
func Sigmoid(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

// SigmoidPrime is the derivative of Sigmoid.
func SigmoidPrime(val float64) float64 {
	return Sigmoid(val) * (1 - Sigmoid(val))
}
