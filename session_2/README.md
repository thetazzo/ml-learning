# Session 2

---

## XOR
```
One is not enough
```

* Let's try to simulate the XOR gate 
* We know
```
XOR:
    0 0 -> 0
    0 1 -> 1
    1 0 -> 1
    1 1 -> 0
```
* One special trait of the XOR gate it that it usually cannot be simulated using only one **neuron**
    * This is becuase XOR is not linearly separable -> XOR has values that are equaly destributed with this I mean that XOR contains two sets of values that are of the same size
    ```
        |---------|
       1|X       O|
        |         |
       0|O       X|
        |_________|
         0       1
    ```
    * Neural networks with only one neuron usually cannot solve problems that are not linearly separable

* We can describe XOR using OR, AND, NAND
    ```
    XOR = (x|y) & ~(x&y)
    ```
* Let's make one neuron that will compute the OR gate the other that will compute the NAND gate
* Let's make the third neuron compute the AND of the outputs of the two previous neurons
* The result of this sequence is XOR

--- 

## What is forwarding?
* forwarding is basically a transition from neurons form the previous layer into the neurons of the next layer
* forwarding is used when calculating output values of the neuron 

--

## Finite Difference ~ Gradient 
* Let's try a new approach of computing finite differanceses
* As we encounter more and more complex models more data needs to be calculated
* We can intorduce a `gradient model`
* This is a model has the same structure as the input model but inside it we will store the differances ~ the gradient
* The `gradient model` here represents all the values we need to substract from the original model to drive the function towards the `minimum`

### How will it learn?
* before each iteration of the training process we create a `gradient model`
* The training process consists of taking our `neural network model` and applying to it the `gradient model`
* In this scenario applying signefies **substracting all values of the gradient model (multiplied by the learning rate) from the neural network model**
