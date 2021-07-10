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
    '''prints out the weight matrixes W[0], W[1] and W[2]'''
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
    '''returns the network output of a specific input specifically.
    NOTE:input must be given without bias input'''
    global W
    p=lambda z:expit(np.matmul(np.pad(x,((0,0),(1,0)),
        'constant',constant_values=1),W[z].T))if z==0 else expit(np.matmul(
            np.pad(p(z-1),((0,0),(1,0)),
                   'constant',
                   constant_values=1),
                   W[z].T))
    return p(len(W)-1)


def train(x,y,iterations,learningrate=0.1,plot=False,printy=True,printw=True,
          plot_delay=0.00000001):
  '''over the iterations, this function optimizes the values of W[0],W[1] and
   W[2] to reduce output error.
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
  result=[[] for i in range(len(W))]
  #Lsum=[[] for i in range(len(W))]
  if plot:
      nnplotter.plotinit()
  p=lambda z:expit(np.matmul(np.pad(x,((0,0),(1,0)),
        'constant',constant_values=1),W[z].T))if z==0 else expit(np.matmul(
            np.pad(p(z-1),((0,0),(1,0)),'constant',constant_values=1),W[z].T))
  for j in range(iterations):
      for i in range(len(W)-1,-1,-1):
          result[i]=p(i)
          if i !=len(W)-1:
              result[i]=np.pad(result[i],((0,0),(1,0)),'constant',
                    constant_values=1)
      for i in range(len(x)):
          for j in range(len(W)-1,-1,-1):
              X=np.pad(x[i],((1,0)),'constant',constant_values=1)
              Wcorr[2]=np.matmul(np.array([(result[2][i]-y[i])*(result[2][i]*(
                      1-result[2][i]))]).T,np.array([result[1][i]]))
              Wcorr[1]=np.matmul((np.matmul(np.array([(result[2][i]-y[i])*(
                  result[2][i]*(1-result[2][i]))]),W[2])*np.array([(
                      result[1][i]*(1-result[1][i]))])).T,np.array(
                          [result[0][i]]))
              Wcorr[0]=np.matmul((np.matmul((np.matmul(np.array([(
                  result[2][i]-y[i])*(result[2][i]*(1-result[2][i]))]),
                      W[2])*np.array([(result[1][i]*(
                          1-result[1][i]))]))[0][1:],W[1])*np.array([(
                              result[0][i]*(1-result[0][i]))])).T,[X])
              W[2]=W[2]-learningrate*Wcorr[2]
              W[1]=W[1]-(learningrate/2)*np.delete(Wcorr[1],0,0)
              W[0]=W[0]-(learningrate/4)*np.delete(Wcorr[0],0,0)
      if plot:
          nnplotter.ax.clear()
          for i in range(len(W)):
              nnplotter.plotweights(W[i],i)
          nnplotter.plt.pause(plot_delay)
      if printy:
          print(predict(x))
  if printw:
    for i in range(len(W)):
        print('W[%d]=\n'%i,W[i],'\n')
  return

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
    except Exception:
        print('Something went wrong double check {} exist in current \
              working directory'.format(filename))