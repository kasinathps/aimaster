Contents:  
========
nn
--

### nn		:: Always updated to the Latest Neural Network Model of aimaster.

#### Current Features of nn::

- Model Class with Initialization.
- Neural Network visualization.
- Could specify Activation ("relu" or "sigmoid".Other activations are in development)
- Live training with visualization. ("Pretty Slow 😁").Training without visualization prefered(fast).
- Saving and Loading Models. (Using pickle)


### nn1hlnb :  
 Neural Network with '1' Hidden Layer and No Bias.  
### nn1hlib :  
 Neural Network with '1' Hidden Layer with input Bias.  
### nn1hlhb :  
 Neural Network with '1' Hidden Layer with hidden layer Bias.  
### nn1hlfb :  
 Neural Network with '1' Hidden Layer with full bias(both input and Hidden layer Bias).  
### nn2hlfb:  
 Neural Network with '2' Hidden Layer with full bias.  
### nnfb:  
 Neural Network with adjustable architecture full biased.

```python
nn.createnn(["your desired architecture"]
```
Example ```nn.createnn([4,5,5,3])``` gives a 
neural network with 4 inputs plus a bias neuron and two hidden layers of 5 neurons each and with bias and an output layer of 3 neurons.
  
### tools ::  
 contains aimaster's tools like visualization functions.
