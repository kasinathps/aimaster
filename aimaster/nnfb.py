'''This module uses Sigmoid activation only'''
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
    global W,arch
    arch=architecture
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
    '''returns the network output of a specific input specifically
        NOTE:
            if a single input is given to predict funcion it must be of shape
            (1,n) {n = no of input neurons}  
            Example: [1 , 1] has a shape of (2,) which is not accepted yet
            but [[1, 1]] has a shape of (1,2) which is desired if single input
            You know what you are doing :) '''
    global W
    p=lambda z:expit(np.matmul(np.pad(x,((0,0),(1,0)),
        'constant',constant_values=1),W[z].T))if z==0 else expit(np.matmul(
            np.pad(p(z-1),((0,0),(1,0)),
                   'constant',
                   constant_values=1),
                   W[z].T))
    return p(len(W)-1)


def savemodel(filename):
    np.savez(filename,arch,W)


def loadmodel(filename):
    global arch,W
    if '.npz' not in filename:
        a=np.load(filename+'.npz')
        arch,W=[a[i] for i in a.keys()]
    if '.npz' in filename:
        a=np.load(filename)
        arch,W=[a[i] for i in a.keys()]
    try:
        print('Model Architecture = {}\n'.format(arch))
        print('Model Weights W = \n {}'.format(W))
    except:
        print('Something went wrong double check {} exist in current \
              working directory'.format(filename))


def train(x,y,iterations,learningrate=0.1,plot=False,Plotfreq=1,Plotmaxm=0,printy=True,printw=True,plot_delay=0.00000001):
  '''over the iterations, this function optimizes the values of all weights
  to reduce output error.
  if printy is set True (default) it prints the output on each iterations,
  if printw is set True (default) it prints the weight matrices on the 
  end of iterations 
  
  NOTE: learning rate is set default to 0.1 which sometimes is
  morethan required (result: gradient descent will not converge) or otherwise
  less than required (result: slow training). So feel free to experiment with 
  different values as this module is for basic understanding and experiments :)
  '''
  global W
  Wcorr=W*0
  result=[[] for i in range(len(W))]
  #Lsum=[[] for i in range(len(W))]
  if plot:
      nnplotter.plotinit()
  p=lambda z:expit(np.matmul(np.pad(x,((0,0),(1,0)),
        'constant',constant_values=1),W[z].T))if z==0 else expit(np.matmul(
            np.pad(p(z-1),((0,0),(1,0)),'constant',constant_values=1),W[z].T))
  for k in range(iterations):
      for i in range(len(W)-1,-1,-1):
          result[i]=np.pad(p(i),((0,0),(1,0)),'constant',constant_values=1)
      for i in range(len(x)):
          X=np.pad(x[i],((1,0)),'constant',constant_values=1)
          for j in range(len(W)-1,-1,-1):
              if j==len(W)-1:
                  Wcorr[j]=np.array([(result[j][i]-y[i])*(result[j][i]*(1-result[j][i]))])
              else:
                  Wcorr[j]=(np.matmul(Wcorr[j+1][0][1:],W[j+1])*np.array([(result[j][i]*(1-result[j][i]))]))
          for j in range(len(W)-1,-1,-1):
              if j==0:
                  W[0]=W[0]-learningrate*np.delete(np.matmul(Wcorr[0].T,np.array([X])),0,0)
              else:
                  W[j]=W[j]-learningrate*np.delete(np.matmul(Wcorr[j].T,np.array([result[j-1][i]])),0,0)
      if plot:
          if k==0 and Plotmaxm:
              figManager = nnplotter.plt.get_current_fig_manager()
              figManager.window.showMaximized()
          if k%Plotfreq == 0:
              nnplotter.ax.clear()
              for i in range(len(W)):
                  nnplotter.plotweights(W[i],i)
              nnplotter.ax.text(0,0,s='iteration {}'.format(k))
              nnplotter.plt.pause(plot_delay)
      if printy:
          print(predict(x))
      print('iteration : {}'.format(k+1))
  if printw:
    for i in range(len(W)):
        print('W[%d]=\n'%i,W[i],'\n')
  return
