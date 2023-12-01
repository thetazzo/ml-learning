# Machine Learning 

``
Exploring the relm of machine learning
``

---

# Session 01 

```
1. Create and run tests
2. Gather data
2. Construct a mathematical model 
3. Tweak around the model to make it work better with the data and move closer to the desired result
4. This tweaked model is a neural network
```
---

## Twice
```
A trivial example, proof of concept
```

* This is an example of an artificial neuron with a single input and single output

* Let's explore the training of a forumla that on a given input ``x`` outputs it's twice ``2*x``:
    ```
        0 -> 0
        1 -> 2
        2 -> 4
        3 -> 6
        4 -> 8
    ```
* We can deduce that the formula has a form like this 
$$f(x) = x * w$$
* ``x`` ... input value
* ``y`` ... multiplication parameter

### Our goal
* Make the parameter `w` converge to `2` so that when used in the $f(x)=x*w$ the function returns the as correct of an output as possible

### Preparation
* Before we can make the computer learn what the optimal value for parameter ``w`` should be we must define some training data
```
[
    {0, 0}
    {1, 2}
    {2, 4}
    {3, 6}
    {4, 8}
]
```
* Training data is list of  ``{input, output}`` pairs which define what output shout the function have for the given input
* Let's initialize the parameter ``w`` as some *random float* value.

## Artificial neurons
* **Artificial neurons** are the basic building blocks of any artificial neural network ~ `elementary units`
* An **artificial neuron** recieves one or more inputs and sums them to produce an output or so called `activation` 
* Usually each input has it's own separate `weight`
* And the sum is often added to a value know as `bias` ~ a value that is not dependent on the input and shifts the sum 
* Usually in the last step we put the sum through a *non-linear function* known as an activation function that produces the output
* Activation functions usually have a **sigmoid shape** but they may also take the form of other non-linear functions, *picewise linear functions* or *step-functions* 
---

### Cost function 
* Let's take a look at a cost function for an **artificial neuron with one connection**
* This function will mesure the correctness of our model
* The function will use our training data and parameter `w` and parameter `b` to calucalte ``train_error`` value with will tell us how far off our model is in regards to the training data (how incorrect our current model in comparison to our training data)
    * `w` parameter represents the weight of our connection 
    * `b ` parameter represents the bias

* For each row the training alghoritem will take the input parameter of the training data as ``x``  and multiply it by the given ``w`` and store the ouput into ``y``
    $$y = x*w + b$$

* Let's calcuate the distance between the desired output of the training data `tdy` and the calcualated output `y`
$$distance = y - tdy$$

* Let's accumulate the squres of `distance` into `train_error`
$$train\_error = train\_error + (distance*distance)$$

* Using squared distances we achieve:
    * **absolute values** ~ we squre the distance is so that we don't have to deal with negative numbers 
    * **aplified errors** ~ we square the distance is so that any large offset from the desired output is amplified  
    
* After we traverse all of the traning data and have accumulated the ``train_error`` we want to normalize the error (find it's average) so we devide it by the amount of training data
$$train\_error = \frac{train\_error}{train\_count}$$

* ``train_error`` is a messure of how badly our model performs
    * the larger the ``train_error`` the worse the model performs
    * the smaller the ``train_error`` the better the model performs

---

* Here we can think about a concept called **OVERFITTING**
    * Overfitting is a scenario where your model fits your data perfectly but fails to predict any new data
    * The model is solely focused on the training data and does not recognies anything outside training data
* In most cases we try to avoid overfitting
* On the other hand this current example ``Twice`` is a trivial example that works on only one parameter and the only correct value for ``w`` is ``2``. This is the reason why overfitting here is not a problem, our training data trains our model to take any input parameter and output it's twice and this is exacly what we wanted to achieve.

---

### What should be the next step?

* Now that we can calcuate the `cost` of our model we wish to minimize it.
    * The closer the `cost` is to 0 the better the moddel behaves on that specific training data (output is closer to the desired output)

* Let's introduce a value `eps`(eplison)
    * This value should be small and it will be used to tweak our parameter `w`
    * Let's use the ``eps`` value to shift our parameter `w`
        $$w = w - eps$$
        $$or$$
        $$w = w + eps$$
    * By shifting the parameter `w` by `eps` we can achive eather an **imprevement** or **decline** in the ``cost`` of our model

### How does it learn?
* Let's try to think about out `cost function` as a mathematical function
    * We know that mathematical functions usually have some sort of `minimum` and/or `maximum` values(points) ~ the lowest / highest value(point) our function can reach.
    * We know that if we take the derivative of our function in some point `x` we can tell in which way our function grows
    * Now that we know in which way our function grows we can move into the opposite direction of it's growth which will result in us moving towards the `minimum` of our function
    * From all of this we can conclude that if we move towards the `minimum` of our `cost function` we are converging the `cost` of our model towards 0

### Finite difference
* With the idea of driving the `cost` of our model to 0 using derivatives let's explore a way of aproximating derivatives using a method named `finite difference`

* A thought about *finite diffrence method*: 
    * We should keep in mind that this method is not used in the realm of neural network engineering because it is **slow** and **inaccurate**
    * In contrast `finite difference` method can be used as the part of the our learning process when trying to understand how neural networks work
$$\$$
* *Let's recap*:  As of right now we are trying to find the `minimum` of the `cost function` by looking in which direction we want to move our parameter `w` so that we reach the `minimum` of our `cost function`

* From the definition of derivatives we know:
    $$L=\lim_{h \to 0}\frac{f(x + h) - f(x)}{h}$$ 
    *A function of a real variable f(x) is differentiable at a point a of its domain, if its domain contains an open interval I containing a, and the limit L exists This means that, for every positive real number $\epsilon$ (even very small), there exists a positive real number $\delta$ such that, for every h such that $|h|<\delta$ and $h \neq 0$ then $f(a+h)$ is defined, and*
    $$|L-\frac{f(a+h)-f(a)}{h}| < \epsilon$$
    Where `| ... |` denote the absolute value

* In other words we take the distance between the result of function `f` shifted by the parameter `h` and the the result of function `f` that is not shifted and we devide this distance by that same value `h`
* We do this as we drive the parameter `h` towords 0.
      
* From the definition of `finite difference` we know:
    $$△_h[f](x)=f(x - h) - f(x)$$

* Let's combine both ideas to compute the error of our cost (``cost_error``)
    $$cost\_error=\frac{(cost(w + eps) - cost(w))}{eps}$$

* Now let's subtract `cost_error` from the parameter `w` so that our `cost function` starts to move towards 0
    $$w = w - cost\_error$$

* The first issue we encounter when computing `cost_error` is that the `cost_error` values appear to be a large number which results in our parameter `w` "jumping around too much" and never reaching the desired value

* Let's introduce the `learining rate` concept to our model 
    * We are now able to have more control over the learning speed of our model
    * In our case it will solve the issue of `cost_error` values being large 
    $$w = w - (lear\_rate * cost\_error)$$

---

