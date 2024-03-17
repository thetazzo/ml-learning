# Session 6

```
Stacking one on top of the other
```

### Gradient Descent Recap
* Computing the derivitive of the `cost function`
* **The Good**
    * We compute the derivitive only once
* **The Bad**
    * On each iteration we compute the **gradient** of the `cost function` of the entire **training dataset**
* **The Ugly**
    * Consequently the **gradient calucation** grows **slower** as the dataset increases and leads to *suboptimal performance*

## Stochastic Gradient Descent ~ SGD

* *Idea: Small batches (collections) of data would take less time to calculate*
* Let's not forget that gradient descent is a good way for our model to train 
* With that we also have to keep in mind that calculations occure on all **training data at once** which leads to growth in *time complexity* as the amount of training data increases 
* To solve this problem and also keep all of the advanages of the gradient descent we should reduce the amount of data on which we perform calculations

* Let's **split** the *training data* into smaller and *mostly equal* batches 
    * Then at radnom choose batches and perform *noisy gradient descent approximation* 
    * Let's use the *approximation* to perform a **descent step**

* This approach allows not only the **time complexity** to be nearly **constant** but also the **memory** used to compute the gradient is also a near **constant** 
* **The Tradeoff**
    * As you may have noticed now we are back to computing **approximations**
    * This leads to each iteration of the gradient descent computation to be less accurate
    * We will tradein **computation accuracy** for **computation speed**
