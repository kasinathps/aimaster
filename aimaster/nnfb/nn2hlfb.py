import numpy as np
from scipy.special import expit
from aimaster.tools import nnplotter
def createnn(architecture=[]):
    if not isinstance(architecture,list) or not architecture:
        print("""give architecture as a list of neurons in each layer including
              input layer,hidden layers and output layer!
              Example: architecture=[2,3,3,1] for nn with input layer of 
                      2 neurons, two hidden layers with 3 neurons each(without 
                      counting bias) and an output layer of 1 neuron""")
        return ValueError
    global W
    W=np.array([np.random.randn(j,i+1) for i,j in zip(architecture[0::1],
                architecture[1::1])])
    for i in range(len(W)):
        print('W[%d]=\n'%i,W[i],'\n')
    return

def weights(plot=False):
    '''prints out the weight matrixes w1 and w2'''
    global W
    if plot:
      nnplotter.plotinit()
      for i in range(len(W)):
          nnplotter.plotweights(W[i],i)
      nnplotter.plt.show()
    for i in range(len(W)):
        print('W[%d]=\n'%i,W[i],'\n')
    return

def predict(x):
    '''returns the network output of a specific input specifically'''
    global W
    p=lambda z:expit(np.matmul(np.pad(x,((0,0),(1,0)),'constant',constant_values=1),
                               W[z].T)) if z==0 else expit(np.matmul(np.pad(p(z-1),
                                ((0,0),(1,0)),'constant',constant_values=1),W[z].T)) 
    return p(len(W)-1)

def train(x,y,iterations,learningrate=0.1,plot=False,printy=True,printw=True,plot_delay=0.00000001):
  '''over the iterations, this function optimizes the values of w1 and
   w2 to reduce output error.
  if printy is set True (default) it prints the output on each iterations,
  if printw is set True (default) it prints the weight matrices on the 
  end of iterations 
  
  NOTE: learning rate is set default to 0.1 which sometimes is
  morethan required (result: gradient descent will not converge) or less 
  than required (result: slow training). feel free to experiment with 
  different values as this module is for basic understanding :)
  '''
  global W
  Wcorr=W*0
  result=[]
  Lsum=[]
  if plot:
      nnplotter.plotinit()
  for j in range(iterations):
      for i in range(len(x)):
          for j in range(len(W)-1,-1,-1):
              Lsum[0]=np.matmul(x[i],W[0].T)
              result[0]=expit(Lsum[0])
              result[0]=np.append([1],result[0])
              Lsum[1]=np.matmul(result[0],W[1].T)
              result[1]=expit(Lsum)
              Wcorr[1]=np.matmul(((result[1]-y[i])*result[1]*(1-result[1])).reshape(-1,1),[result[0]])
              Wcorr[0]=np.matmul(np.array([np.matmul(((result[1]-y[i])*result[1]*(1-result[1])),W[1])*(result[0]*(1-result[0]))]).T,[x[i]])
              W[1]=W[1]-learningrate*Wcorr[1]
              W[0]=W[0]-learningrate*np.delete(Wcorr[0],0,0)
      if plot:
          nnplotter.ax.clear()
          nnplotter.plotweights(W[0],0)
          nnplotter.plotweights(W[1],1)
          nnplotter.plt.pause(plot_delay)
      if printy:
        print('y= ',expit(np.matmul(np.pad(expit(np.matmul(x,W[0].T)),((0,0),(1,0)),'constant',constant_values=(1)),W[1].T)))
  if printw:
    for i in range(len(W)):
        print('W[%d]=\n'%i,W[i],'\n')
  return