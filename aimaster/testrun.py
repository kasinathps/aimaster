import numpy as np
import nn
if __name__ == '__main__':
    X = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    m = nn.model([2,3,1],"sigmoid")
    m.weights(1)
    m.train(X,y,10000,0.1,1,1,1,"queue",0,0.00001)
    m.predict(X)
    m.weights(1)