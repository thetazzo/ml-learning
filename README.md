# Machine Learning 

```
Exploring the relm of machine Learning

1. Create and run tests
2. Gather data
2. Construct a mathematical model 
3. Tweak around the model to make it work better with the data and move closer to the desired result
4. This tweaked model is a neural network
```

---

## TABLE OF CONTENTS

* [Session 1](./session_1/README.md) 
* [Session 2](./session_2/README.md) 
* [Session 3](./session_3/README.md) 

---

## Artificial neurons
* **Artificial neurons** are the basic building blocks of any artificial neural network ~ `elementary units`
* An artificial neuron receives one or more inputs and sums them to produce an output or so-called activation
    * Usually each input has its own separate weight
* And the sum is often added to a value known as bias ~ a value that is not dependent on the input and gives the final result of the sum the needed offset before forwarding it into the activation function
* Usually in the last step we put the sum through a non-linear function known as an activation function that *activates* our neuron 
 * Activation functions usually have a sigmoid shape but they may also take the form of other non-linear functions, picewise linear functions or step-functions


### Why is the bias important?
* You need the bias because the model without it would not be able to fit the data better
* Bias helps you control at which value the activation function will trigger
* If you don't have the bias the model can only modify the output based on the input parameters (weights) but if you introduce bias into the equation the model is capable of shifting the entire state regardless of the input
---

## Finite difference
* With the idea of driving the `cost` of our model to 0 using derivatives let's explore a way of approximating derivatives using a method named `finite difference`

* A thought about the *finite diffrence method*: 
    * We should keep in mind that this method is not used in the realm of neural network engineering because it is **slow** and **inaccurate**
    * In contrast, the `finite difference` method can be used as part of our learning process when trying to understand how neural networks work
$$\$$
* *Let's recap*:  As of right now we are trying to find the `minimum` of the `cost function` by looking in which direction we want to move our parameter `w` so that we reach the `minimum` of our `cost function`

* From the definition of derivatives we know:
    $$L=\lim_{h \to 0}\frac{f(x + h) - f(x)}{h}$$ 
    *A function of a real variable f(x) is differentiable at a point a of its domain if its domain contains an open interval I containing `a` and the limit `L` exists. This means that, for every positive real number $\epsilon$ (even very small), there exists a positive real number $\delta$ such that, for every h such that $|h|<\delta$ and $h \neq 0$ then $f(a+h)$ is defined, and*
    $$|L-\frac{f(a+h)-f(a)}{h}| < \epsilon$$
    Where `| ... |` denotes the absolute value

* In other words we take the distance between the result of function `f` shifted by the parameter `h` and the result of function `f` that is not shifted and we divide this distance by that same value `h`
* We do this as we drive the parameter `h` to 0.
      
* From the definition of `finite difference` we know:
$$△_h[f](x)=f(x - h) - f(x)$$

* Let's combine both ideas to compute the error distance of our cost (`dw`) and the error distance of our bias (`db`)
    $$dw=\frac{(cost(w + eps, b) - cost(w, b))}{eps}$$
    $$db=\frac{(cost(w, b + eps) - cost(w, b))}{eps}$$

* Now let's adjust our parameters by subtracting `dw` from the parameter `w` and `db` from the parameter `b` 
    $$w = w - dw$$
    $$b = b - db$$

* The first issue we encounter when computing error distance values is that values appear to be large numbers which results in our parameters `w`, `b` "jumping around too much" and never reaching the desired values

* Let's introduce the `learning rate` concept to our model 
    * We are now able to have more control over the learning speed of our model
    * In our case it will solve the issue of error distance values being large 
    $$w = w - (lear\_rate * dw)$$
    $$w = w - (lear\_rate * db)$$

---

## Activation function
* After an **artificial neuron** sums up all of its weights and adds its bias the output is forwarded through an **activation function** 
* The goal of the activation function is to take the summed-up data, which can possibly be unbound or  "all over the place", and so to say isolate the value (example between 0 and 1) and make it non-linear.
* One of the standard activation functions is named `Sigmoid`

### Sigmoid
* *It's any mathematical function that has the characteristic "S"-shaped curve or sigmoid curve.* 
* In our case we will look at a sigmoid function that maps values from `-Infinity` to `+Infinity` to values from `0` to `1`
    * The closer you are to `-Infinity` the closer you are to `0`
    * The closer you are to `+Infinity` the closer you are to `1`
* Let's take a look at a common example of a sigmoid function ~ `Logistic function`
$$\sigma(x)=\frac{1}{1+e^{-x}}=\frac{e^{x}}{1+e^{x}}=1-\sigma(-x)$$

--- 

