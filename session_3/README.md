
# Session 3

--- 

## Understanding How Information Is Passed Through A Neural Network

![Image](./how_neural_networks_work.png)

* Let's represent data passing through the neural network as **matrices**
 * Let `A` be the matrix containing values form the previous activation
    * Here we can also think about the `input layer` as an `activation` of *0-th layer* 
    * This allows us to think of the inputs ($x_1, x_2, ..., x_n$) as `activations` of the *0-th layer* ($a_{01}, a_{02}, a_{03}, ..., a_{0n}$) 
 * Let `W` be the matrix containing weights of the connections to the following layer 
    * Each neuron of the layer has it's own corresponding `row` in the matrix `W`
    * Each neuron's row is build from colums and there are as many colums as there are connections going out of the neuron.
        * As of right now we are only talking about neural networks with `fully connected layers` ~ This layers have all neurons between then connected
        * The above reason is why we can assume that all rows will have the same amount of colums
    * Each pair of row and column `W[row][column]` holds a value of the connection to the next layers
* Let `B` be a matrix containing values of all the biases of the neuros of the following layer 
* Let's use $\sigma$ to denote the `activation function` (as an example let's use sigmoid activation function ~ $\sigma$)

## How To Compute A Layer

* Let's assume that our *input layer* has input values denoted as $x_y$ where y is the *index* of the input 
* Let's denote neurons in the following layer as $a_z$ where z is the *index* of the neuron 
* Let the **weight** of the connection from input $x_y$ to neuron in the following layer $a_z$ be denoted as $w_{xz}$  

* Here is how we would compute the activations of the **layer 1** of our example neural network

$$\sigma \left(\begin{bmatrix} x_1 & x_2 \end{bmatrix} \cdot \begin{bmatrix} w_{11} & w_{12} \\\ w_{21} & w_{22} \end{bmatrix} + \begin{bmatrix} b_1 & b_2 \end{bmatrix}\right) = \begin{bmatrix} a_1 & a_2 \end{bmatrix}$$

### How would we compute such an equation
$$\begin{bmatrix} x_1 & x_2 \end{bmatrix} \cdot \begin{bmatrix} w_{11} & w_{12} \\\ w_{21} & w_{22} \end{bmatrix}$$
1. The first step is to compute matrix multiplication between the `activations matrix` ~ `A` and `weights matrix` ~ `W`
    * For the multiplication between matrices to be possible we must ensure that the amount of colums in the first matrix is equal to the amout of rows of the second matrix
    * Let's consider matrix `A` to be of order `a x b` and matrix `W` to be of order `b x c`
    * Let's consider the output matrix as $C = A \times B$ which is going to be of order `a x c`
    * Then an element in matrix `C` is defined as:
    $$C_{ij} = A_{i1}B_{j1} + ... + A_{ib}B_{jc} = \sum_{k=1}^{b}a_{ik}b_{kj}$$
    * for $i=1, ..., a$ and $j=1, ..., c$
---
$$\begin{bmatrix} (x_1 \cdot w_{11} + x_2 \cdot w_{21}) & (x_1 \cdot w_{12} + x_2 \cdot w_{22}) \end{bmatrix} + \begin{bmatrix} b_1 & b_2 \end{bmatrix}$$

2. The second step is to perform matrix addition between the resulting matrix form the first step and the `biases matrix` ~ `B`
    * Here we must ensure that both matrices are of the same *order* this means that both matices have the same amount of rows and columns
    * The sum output matrix is produced by summing up all of the corresponding terms in the matrices
---

3. The last step is to forward our resulting matrix through the activation function of our neural network
* As the output we get a matrix of `avtivations` which we use as the **input of the next layer**

$$\begin{bmatrix}a_1 & a_2 & a_3 & ... & a_n \end{bmatrix}$$

---

* Example of computing the **next layer**

$$\sigma \left(\begin{bmatrix} a_1 & a_2 \end{bmatrix} \cdot \begin{bmatrix} w_{31} \\\ w_{41} \end{bmatrix} + \begin{bmatrix} b_3 \end{bmatrix}\right) = \begin{bmatrix} a_3 \end{bmatrix}$$

* Here we take the activations $a_1$ and $a_2$ form the previous layer and multiply them with the single neuron of the final layer and sum the result with the bias of the last layer 
* As an output we get `activation` $a_3$ which in this case is also the output of out neural network
---

### A Thought About The Input layer
* Form some of the definitions of how to compute layers of neural networks we could make an observation that the *input layer* could be named the **0th layer** or **layer 0** 
* Furthermore we can deduce that **input values** $x_0$, $x_1$, ... $x_y$ are in fact `activations` of the **0th layer** $a_{01}$, $a_{02}$, ..., $a_{0y}$

---

## XOR Implementation With NeuralFramework

---

### Description Of The Neural Model
* Let's think of a possible description that would encapsulate all the data needed to represent a neural model
    ```
    NeuralNetworkModel = {
        count,
        weights[],
        biases[],
        activations[]
    }
    ```
    * Here we use `count` as the descriptor value of the amount of layers our neural network model will have 
    * We use `weights[]` as the descriptor of an array of *matrices* representing all *weights* of connections in our model
    * Similarly we use `biases[]` as the descriptor of an array of *matrices* representing all *biases* in our model
    * Lastly we use `activations[]` as the descriptor of an array of *matrices* representing all *activations* of out model
        * Here we should note that the amount of activations is equal to `count+1`
        * The reason for this is that we define the *input layer* as the **0th activation**  

---

### How Can A Computer Construct A Neural Network Model
* Let's define a function that will accept as inputs an integer array `architecture` and a integer `arch_count` 
    * `architecture` array defines, as the name suggests, the architecture of our model 
        * it is defined as an array of integers where each integer represents the amount of neurons of a layer
        ```
        [2, 2, 1]
        ```
        * Represents an architecture of a neural network where:
            * the **0th layer** or *input layer* has `2` inputs
            * the **1st layer** has `2` *activations* ~ has `2` *neurons* 
            * the **2nd layer** or in this case the *output* has `1` *activation* ~ has `1` *neuron* ~ `1` output 
    * The function will handle the allocation of memory of storing all the matrix representations 

---

* The learning process of our model will occure in two steps:
    1. Calculationg a `gradient neural network` using the `finite difference` method 
    2. Applying the `gradient neural network` to our `neural network`

---

### Computing Finite Difference Using Matrices ~ Gradient Neural Network

* We are still using the same principle method of `finite difference` to optimize (decrease) our `cost function`  
    * Out `cost function` still takes the the output of our current neural network and calculates the distance (*error*) from expected output in our *training data*
* Our model stores all **weights** and **biases** as array data 
* Let's traverse all layers of our neural network and build a new new neural network (`gradient neural network`) that will store the distances (*errors*) of our neural network 
    * To recap how we calcualte the distance (*error*) for a single value:
        1. We compute the cost of the neural network ~ `originalCost`
        2. Take the value (ex. weight of bias) and tweak it by adding to it a value $\varepsilon$
        3. compute the `newCost` cost of the neural network and from it substract the `originalCost` cost of the neural network then devide the result by the same value $\varepsilon$
        $$\frac{(newCost - originalCost)}{\varepsilon}$$
    * The resulting `gradient neural network` will store distances (*erros*) corresponding to all *weights* and *biases* of the neural network

---

### Applying The Gradient Network 
* After calculating the gradient network applying it to the neural network is quite straight forward
* Again let's treverse all the layers of our neural network and from each *weight* and *bias* of our neural network let's substract the corresponding distance (*error*) in the provided `gradient network` multiplied by the provided `learning rate`
    $$W_{ij} = W_{ij} - (learningRate \ \cdot W^g_{ij})$$
    * $W_{ij}$ is a value in the *weights* matrix of the neural network at i-th row and j-th column
    <br></br>
    * $W^g_{ij}$ is a value in the *weights* matrix of the `gradient neural network` at i-th row and j-th column
 
