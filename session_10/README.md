
# Further optimization - STOCHASTIC GRADIENT DESCENT
* The problem I encountered with the amount of training data increasing is connected to the computation of
the ``cost function``.
* The ``cost function`` requires me to iterate through all of the training data samples to compute the cost. 
This leads to longer computation times with the amout of data increasing.
* We can overcome this by introducing **STOCHASTIC GRADIENT DESCENT**

## What is STOCHASTIC GRADIENT DESCENT?
* Stochastic gradient descent is the process of splitting the training data into smaller preferably equal size 
batches and then shuffeling them randomly. 
* This enables you to insted of training on the whole dataset to train one one single batch at a time until
you trained on all batches
* We can think about this like so: Instead of descending streight to the local miniumum you descend into random 
directions and because you move through all of the training batches you in theory on average still converge
to the local minimum ~ instead of going straight to the local minimum you jump around and on average you still
converge to the local minimum

## What is the benefit?
* Stochastic gradient descent in theory achieves the same result as non-stochastic gradent descent with the 
benefit of being less computationally intensive because you don't have to compute on all of the data 

## Implementation

### Training data shuffle
* We need to shuffle the rows of our training data matrix 
* Let's use the Fisher-Yates Shuffle alghoritm for shuffling matrix rows
    * The alghoritem works on the principle of putting all the items into a hat and drawing them out one by
    one randomy and arange the drawn items in a sequence you drew them out
* The alghoritem can be implemented by having a list of items and a pointer to one of the items. The pointer 
can be thought of as a separator to what is in the hat and hat has been already taken out of the hat. 
Everything on the right side of the pointer is still in the hat and everything on the left side has already
been taken out. The "shiffle" step consists of picking a random element from the list of elements that are on 
the right side of the pointer and swaping the picked element with the element the pointer points to. Then move 
the pointer one position to the right. Repeat this process until you react the end of the list.

### Batches
* Let's define the amount of *batches per frame* as ``bpf``. This value will dictate how many batches of 
training data we should process in a single render frame.
* Let's define ``batch_size`` witch will dictate how big should one batch be. For now let's set it to 28. 
I've chosen this value because of the image size being 28x28.
* Let's also define ``batch_count`` which reprensents the total amount of batches 
    * Let's define it as the number of rows of the training data ``td.rows`` devied by the ``batch_size``
    * Let's add ``batch_size-1`` to the number of traing data rows and solve the rounding of the value
    * If there is any remainder it is going to overflow and the division is going to be +1, if there is no
    remeinder there will be no overflow
    $$batch\_count = \frac{td.rows + batch\_size-1}{batch\_size}$$
* Let's introduce ``batch_begin`` which helps us track the position of the begging of the batch
    * Let's start at ``0`` ~ the start of the first batch
    * We also know the size of the batch (``batch_size``) we have to process, so after processing a batch
    let's increment ``batch_begin`` by the ``batch_size`` so that in now points to the next batch
    * Let's repeat this process until we process all the batches 
    * If there is a small remainder it is easily detectable beacuse the current 
    ``batch_begin + batch_size`` overflows the amout of batches so we can easily cut down the size of the last
    batch
* As stated before ``batch_begin + batch_size`` helps us detect the overflow of the amount of batches which 
helps us detect the last batch and allows us to handle it as a special case
    * After computing the last batch we want to compute the *average cost* and add it to the *cost plot*

### Computing The Cost
* Let's compute the cost for each batch as $C_1, C_2, C_3 ...$
* After computing all the costs let's sum them up and devide them by the ``batch_count``
$$cost=\frac{\sum_{i=1}^{batch\_count} C_{i}}{batch\_count}$$
    
---

* stb_image: [github](https://raw.githubusercontent.com/nothings/stb/master/stb_image.h)
* stb_image_write: [github](https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h)
* mnist images: [github](https://github.com/myleott/mnist_png/tree/master)
