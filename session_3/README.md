
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
* Let's use $\sigma$ to denote the `activation function` (in this example we will use the sigmoid activation function ~ $\sigma$)
