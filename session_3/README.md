
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

### How would we compute such an equation:
$$\begin{bmatrix} x_1 & x_2 \end{bmatrix} \cdot \begin{bmatrix} w_{11} & w_{12} \\\ w_{21} & w_{22} \end{bmatrix}$$
1. The first step is to compute matrix multiplication between the `activations matrix` ~ `A` and `weights matrix` ~ `W`
    * For the multiplication between matrices to be possible we must ensure that the amount of colums in the first matrix is equal to the amout of rows of the second matrix
    * The resulting matrix will have as many rows as the first matrix and the amount of colums as the second matrix
    * Each value in each row of the first matrix we multiply with each value in each column in the second matrix and then we sum them up
    $$c_{ij} = \sum_{k=1}^{n}a_{ik}b_{kj}$$
    * were n is the abmount of columns in the second matrix
    * where $c_{ij}$ is the output entry of the product of the i-th row of the first matrix and j-th column of the second matrix
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
