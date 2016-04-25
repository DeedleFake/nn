package nn_test

import (
	"github.com/DeedleFake/nn"
	"testing"
)

func TestNetwork(t *testing.T) {
	n := nn.New(3, 5, 2)
	t.Log(n.Run(3, 2, 1))
}
