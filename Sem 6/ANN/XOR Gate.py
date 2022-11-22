class MpNeuron:
    def __init__(self):
        self.inputsAndOutputs = []
        self.weights = []
        self.threshold = []

    def additionInputOutput(self, inp, out):
        self.inputsAndOutputs.append([inp, out])

    def getNetInputs(self, inp, weights):
        netinp = 0
        for i in range(len(inp)):
            netinp += inp[i] * weights[i]
        return netinp

    def conditionCheckWeightsSet(self, weights):
        threshold = 10000000
        # print(weights)

        for inp, out in self.inputsAndOutputs:
            if out == 1:
                threshold = min(threshold, self.getNetInputs(inp, weights))

        for inp, out in self.inputsAndOutputs:
            if out == 0 and threshold <= self.getNetInputs(inp, weights):
                return False

        self.weights = weights
        self.threshold = threshold
        return True

    def calculationForValidWeight(self, low, high):
        for w1 in range(low, high + 1):
            for w2 in range(low, high + 1):
                if self.conditionCheckWeightsSet([w1, w2]):
                    return True
        return False

    def getOutputCalculations(self, inp):
        if self.getNetInputs(inp, self.weights) >= self.threshold:
            return 1
        return 0


andNotNeuron = MpNeuron()
andNotNeuron.additionInputOutput([0, 0], 0)
andNotNeuron.additionInputOutput([0, 1], 0)
andNotNeuron.additionInputOutput([1, 0], 1)
andNotNeuron.additionInputOutput([1, 1], 0)
andNotNeuron.calculationForValidWeight(-1, 1)
print('For ANDNOT, ([w1, w2], threshold): ',
      andNotNeuron.weights, andNotNeuron.threshold)
print('x1', 'x2', 'out')
for i1 in range(0, 2):
    for i2 in range(0, 2):
        print(i1, '', i2, '', andNotNeuron.getOutputCalculations([i1, i2]))
print()

andNotReverseNeuron = MpNeuron()
andNotReverseNeuron.additionInputOutput([0, 0], 0)
andNotReverseNeuron.additionInputOutput([0, 1], 1)
andNotReverseNeuron.additionInputOutput([1, 0], 0)
andNotReverseNeuron.additionInputOutput([1, 1], 0)
andNotReverseNeuron.calculationForValidWeight(-1, 1)
print('For reverse of ANDNOT, ([w1, w2], threshold): ',
      andNotReverseNeuron.weights, andNotReverseNeuron.threshold)
print('x1', 'x2', 'out')
for i1 in range(0, 2):
    for i2 in range(0, 2):
        print(i1, '', i2, '', andNotReverseNeuron.getOutputCalculations([i1, i2]))
print()

orNeuron = MpNeuron()
orNeuron.additionInputOutput([0, 0], 0)
orNeuron.additionInputOutput([0, 1], 1)
orNeuron.additionInputOutput([1, 0], 1)
orNeuron.additionInputOutput([1, 1], 1)
orNeuron.calculationForValidWeight(-1, 1)
print('For OR, ([w1, w2], threshold): ',
      orNeuron.weights, orNeuron.threshold)
print('x1', 'x2', 'out')
for i1 in range(0, 2):
    for i2 in range(0, 2):
        print(i1, '', i2, '', orNeuron.getOutputCalculations([i1, i2]))
print()

print('')
print('XOR Gate Neuron:')
print('x1', 'x2', 'out')
for i1 in range(0, 2):
    for i2 in range(0, 2):
        print(i1, '', i2, '', orNeuron.getOutputCalculations(
            [andNotNeuron.getOutputCalculations([i1, i2]), andNotReverseNeuron.getOutputCalculations([i1, i2])]))

print('Hence, XOR Gate can be realised using MP Neuron')