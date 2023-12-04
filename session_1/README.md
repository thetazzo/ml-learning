
# Session 01 

---

## Twice
```
A trivial example, proof of concept
```

* This is an example of an artificial neuron with a single input and a single output.

* Let's explore training a forumla so that for a given input `x` it outputs twice `2*x`:
    ```
        0 -> 0
        1 -> 2
        2 -> 4
        3 -> 6
        4 -> 8
    ```
* We can deduce that the formula has the form 
$$f(x) = x * w$$
* `x` ... input value

### Our goal
* To make the parameter `w` converge to `2`, so that when used in $f(x)=x*w$, the function gives as correct an output as possible.

### Preparation
* Before we can make the computer learn what the optimal value for the parameters `w`, `b` should be, we need to define some training data.
```
[
    {0, 0}
    {1, 2}
    {2, 4}
    {3, 6}
    {4, 8}
]
```
* Let's initialise the parameter `w`, `b` as any *random float* value.

## Cost function ~ Artificial neuron with one connection 
* This function will measure the correctness of our model.
* The function will use our training data and the parameter `w` and parameter `b` to calculate the `cost` of our model which will tell us how far off our model is in relation to the training data (how incorrect our current model is in relation to our training data).
    * The `w` parameter represents the weight of our connection. 
    * The `b` parameter represents the bias.

* For each row, the training algorithm will take the input parameter of the training data as `x` and multiply it by the given ``w`` and add the bias parameter `b`` to the multiplication.
* Note: We are not using any activation functions here (bad idea).
* Store the final result in `y`

    $$y = x*w + b$$

* Let's calculate the differance between the desired output of the training data `tdy` and the calculated output `y`.
    
    $$differance = y - tdy$$

* Let's accumulate the squares of `difference` in `cost`
    
    $$cost = cost + (difference*difference)$$

* Using squared differences we get
    * **absolute values** ~ we square the differance so that we don't have to deal with negative numbers. 
    * Amplified errors** ~ we square the differance so that any large offset from the desired output is amplified.  
    
* After we've traversed all the training data and accumulated the `cost`, we want to normalise the error (find it's average), so we divide it by the amount of training data.
    $$cost = \frac{cost}{trainCount}$$

* The `cost` is a measure of how badly our model is performing.
    * The larger the `cost`, the worse the model's performance.
    * The smaller the `cost`, the better the model performs.

---

* Here we can think of a concept called **OVERFITTING**.
    * Overfitting is a scenario where your model fits your data perfectly, but fails to predict new data.
    * The model is only focused on the training data and does not recognise anything outside the training data.
* In most cases we try to avoid overfitting.
* On the other hand, this current example `Twice` is a trivial example that works on only one parameter and the only correct value for `w` is `2`. This is the reason why overfitting is not a problem here, our training data trains our model to take any input parameter and output it twice, and that's exactly what we wanted to achieve.

---

### What should be the next step?

* Now that we can calculate the `cost` of our model, we want to minimise it.
    * The closer the `cost` is to 0, the better the model will perform on that particular training data (the output will be closer to the desired output).

* Let's introduce a value `eps` (eplison).
    * This value should be small and will be used to optimise our parameter `w`.
    * Let's use the value of `eps` to move our parameter `w`
        $$w = w - eps$$
        or
        $$w = w + eps$$
    * By shifting the parameter `w` by `eps`, we can either get an **improvement** or **decrease** in the ``cost`` of our model.

### How does it learn?
* Let's try to think of our cost function as a mathematical function.
    * We know that mathematical functions usually have some sort of `minimum` and/or `maximum` values (points) ~ the lowest / highest value (point) our function can reach.
    * We know that if we take the derivative of our function at some point `x`, we can see in which direction our function is growing.
    * Now that we know in which direction our function is growing, we can move in the opposite direction of it's growth, which will result in us moving towards the `minimum` of our function.
    * From all this we can conclude that if we move towards the `minimum` of our `cost function`, we will converge the `cost` of our model to 0.

### Finite difference
* With the idea of using derivatives to drive the `cost` of our model to 0, let's explore a way of approximating derivatives using a method called `finite difference`.

* A thought about the `finite difference method`: 
    * We should remember that this method is not used in the field of neural networks because it is **slow** and **inaccurate**.
    In contrast, the *finite difference* method can be used as part of our learning process when trying to understand how neural networks work.
* Let's recap:  Right now we are trying to find the `minimum` of the `cost function` by looking in which direction we want to move our parameter `w` so that we get the `minimum` of our `cost function`.

* From the definition of derivatives we know that
    $$L=\lim_{h \to 0}\frac{f(x + h) - f(x)}{h}$$ 
    *A function of a real variable $f(x)$ is differentiable at a point `a` of its domain if its domain contains an open interval I containing `a`, and the limit `L` exists. This means that for every positive real number $\varepsilon$ (even very small ones), there exists a positive real number $\delta$ such that, for every h such that $|h|<\delta$ and $h \neq 0$, then $f(a+h)$ is defined, and* 

    $$\lvert L-\frac{f(a+h)-f(a)}{h}\rvert < \varepsilon$$
    
    where `|...|` define an absolute value

* In other words, we take the differance between the result of the function `f` shifted by the parameter `h` and the result of the function `f` not shifted, and we divide this differance by the same value `h`.
* We do this by setting the parameter `h` to 0.
      
* From the definition of `finite difference` we know that
$$\Delta_h[f](x)=f(x - h) - f(x)$$

* Let's combine both ideas to calculate the error differance of our cost (`dw`) and the error differance of our bias (`db`).
    $$dw=\frac{cost(w + eps, b) - cost(w, b)}{eps}$$
    $$db=\frac{cost( w, b + eps) - cost(w, b)}{eps}$$

* Now let's adjust our parameters by subtracting `dw` from the parameter `w` and `db` from the parameter `b`. 
    $$w = w - dw$$
    $$b = b - db$$

* The first problem we encounter when calculating error differance values is that the values appear to be large numbers, causing our parameters `w`, `b` to "jump around" too much and never reach the desired values.

* Let's introduce the concept of `learning rate (rate)` into our model.
    * We can now have more control over the learning rate of our model.
    * In our case it will solve the problem of large error differance values 
    $$w = w - (rate * dw)$$
    $$b = b - (rate * db)$$

---

## AND & OR gates

```
Example that simualtes logic gates (AND, OR)
```
* This is an example of an artificial neuron with a two inputs and single output

* Let's explore the training of a forumla that takes inputs `x1` and `x2` outputs the value of the gate `x1 AND x2` or `x1 OR x2`:
```
AND:
0 AND 0 -> 0
1 AND 0 -> 0
0 AND 1 -> 0
1 AND 1 -> 1

OR:
0 OR 0 -> 0
1 OR 0 -> 1
0 OR 1 -> 1
1 OR 1 -> 1
```
* We can deduce that the formula has a form like this 
$$f(x) = x1 * w1 + x2*w2$$
* `x1` ... first input value
* `x2` ... second input value

### Our goal
* Wind the optimal parameters `w1`, `w2` and `b` that will correctly evalue the gate operation 

### Preparation
* Before we can make the computer learn what the optimal value for parameter `w1`, `w2`, `b` should be we must define some training data
```
   AND:
   [
        {0, 0, 0}
        {0, 1, 0}
        {1, 0, 0}
        {1, 1, 1}
   ]

   OR:
   [
        {0, 0, 0}
        {1, 0, 1}
        {0, 1, 1}
        {1, 1, 1}
   ]
```
* Let's initialize the parameter `w1`, `w2`, `b` as some *random float* value.

---

## Cost function ~ Artificial Neuron With Two Connections 

* This function will mesure the correctness of our model
* The function will use our training data and parameter `w1`, `w2` and parameter `b` to calucalte `cost` value with will tell us how far off our model is in regards to the training data (how incorrect our current model in comparison to our training data)
    * `w1` parameter represents the weight of our first connection 
    * `w2` parameter represents the weight of our second connection 
    * `b ` parameter represents the bias

* For each row the training alghoritem will take the input parameters of the training data as `x1` and `x2` and multiply them by the coresponding `w1` and `w2` then let's add to the sum of multiplications the bias parameter `b` 
* Next step is to forward the whole sum through the activation function (sigmoid)
* Store the final result into `y`

    $$y = sigmoid(x1*w1 + x2*w2 + b)$$

* Let's calcuate the differance between the desired output of the training data `tdy` and the calcualated output `y`

    $$differance = y - tdy$$

* Let's accumulate the squres of `differance` into `cost`

    $$cost = cost + (differance*differance)$$

* Using squared differencess we achieve:
    * **absolute values** ~ we squre the differance is so that we don't have to deal with negative numbers 
    * **aplified errors** ~ we square the differance is so that any large offset from the desired output is amplified  
    
* After we traverse all of the traning data and have accumulated the `cost` we want to normalize the error (find it's average) so we devide it by the amount of training data
    $$cost = \frac{cost}{trainCount}$$

* `cost` is a messure of how badly our model performs
    * the larger the `cost` the worse the model performs
    * the smaller the `cost` the better the model performs


