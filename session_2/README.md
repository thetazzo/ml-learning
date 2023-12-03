# Session 2

---

## XOR
```
One is not enough
```

* Let's try to simulate the XOR gate 
```
XOR:
   0 0 | 0
   0 1 | 1
   1 0 | 1
   1 1 | 0
```
* A special feature of the XOR gate is that it cannot usually be simulated with only one **neuron**.
    * This is because XOR is not linearly separable -> XOR has values that are equally distributed, by this I mean that XOR contains two sets of values that are equally large.
    ```
        |---------|
       1|X       O|
        |         |
       0|O       X|
        |_________|
         0       1
    ```
    * Neural networks with only one neuron cannot usually solve problems that are not linearly separable.

* We can describe XOR with OR, AND, NAND
    ```
    XOR = (x|y) & ~(x&y)
    ```
* Let's make one neuron to do the **OR** gate and another to do the **NAND** gate.
* Let's make the third neuron calculate the **AND** of the outputs of the previous two neurons.
* The result of this sequence is **XOR**

--- 

## What is forwarding?
* **Forwarding** defines how a neural network model processes input data
* Values that are *forwarded* are those values that are put through the neural network model

---

## Finite Difference As "Gradient" 
* Let's try a new approach to computing finite differences.
* As we encounter more complex models, more data needs to be calculated.
* We can introduce a `gradient model`.
* This is a model that has the same structure as the input model, but inside it we will store the differences ~ the gradient.
* The `gradient model` here represents all the values we need to subtract from the original model to drive the function towards the `minimum`.

### How will it learn?
* Before each iteration of the training process, we create a `gradient model`.
* The training process consists of taking our `neural network model` and applying the 'gradient model' to it.
* In this scenario, applying means **subtracting all values of the gradient model (multiplied by the learning rate) from the neural network model**.
