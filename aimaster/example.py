from aimaster.nn import model
from numpy import array
print("use 'example.run()' for demonstration when importing inside python interactive terminal")
def run():
    x=array([[0,0],[0,1],[1,0],[1,1]])
    print("x=array([[0,0],[0,1],[1,0],[1,1]])\n input matrix x \n",x)
    input("Press Enter key to continue...")
    y=array([[0],[1],[1],[0]])
    print("y=array([[0],[1],[1],[0]]) \n output matrix y\n", y)
    input("Press Enter key to continue...")
    M=model([2,3,1])
    print("M=model([2,3,1]) \n Model initialized with 2 input neurons 3 hidden \n\
        neurons( 1 hidden layer) and 1 output neuron. Weights are printed above\n")
    input("Press Enter key to continue...")
    print("M.train(x,y,10000,0.1,1) \n Begining training with default learning rate 0.1 and\
         10000 iterations with visualization enabled")
    input("Press Enter key to start Training!!! This might take some seconds...")
    M.train(x,y,10000,0.1,1)
    print("Training completed.\n M.weights(1) output weight matrix with visualization(1) enabled")
    input("Press Enter key to show weights")
    M.weights(1)
    print("M.predict(x) \nUse the model to predict output for input x [[0,0],[0,1],[1,0],[1,1]]")
    print(M.predict(x))
    input("Press Enter key to exit...(Power Button also does the job)\n")

if __name__=="__main__":
    run()