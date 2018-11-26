import numpy as np
from scipy.special import expit
from aimaster.tools import nnplotter
def createnn(input_size,hidden_layer_size,output_size,pt=1):
    '''creates weight matrixes w1 and w2 ,where w2 holds bias weights
    according to input_size,hidden_layer_size and output_size
    
    NOTE: hidden_layer_size without addressing bias must be given
          bias neuron will be added by the program and weights are created
          accordingly
          Input bias will be added too
          Example: if hidden_layer_size=2 the total hidden layer size will be
          3'''
    global w1,w2
    w1=np.random.rand(hidden_layer_size,input_size+1)
    w2=np.random.rand(output_size,hidden_layer_size+1)
    if pt:
        print('w1:\n',w1,'\nw2:\n',w2)
    return

def weights(plot=False):
    '''prints out the weight matrixes w1 and w2'''
    global w1,w2
    if plot:
      nnplotter.plotinit()
      nnplotter.plotweights(w1,0)
      nnplotter.plotweights(w2,1)
      nnplotter.plt.show()
    print('w1:\n',w1,'\nw2:\n',w2)
    return

def predict(x,add_bias=0):
    '''returns the network output of a specific input specifically'''
    global w1,w2
    if add_bias:
        x=np.pad(x,((0,0),(1,0)),'constant',constant_values=1)
    return expit(np.matmul(np.pad(expit(np.matmul(x,w1.T)),((0,0),(1,0)),
                                  'constant',constant_values=(1)),w2.T))

def train(x,y,iterations,learningrate=0.1,printy=True,printw=True,plot=False,plot_delay=0.00000001):
  
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
  global w1,w2
  x=np.pad(x,((0,0),(1,0)),'constant',constant_values=1)
  if plot:
      nnplotter.plotinit()
  for j in range(iterations):
    for i in range(len(x)):
      Hsum=np.matmul(x[i],w1.T)
      result1=expit(Hsum)
      result1=np.append([1],result1)
      sum=np.matmul(result1,w2.T)
      result=expit(sum)
      e0=result-y[i]
      dtotal_dsum=e0*result*(1-result)
      w2corr=np.matmul(dtotal_dsum.reshape(-1,1),[result1])
      w1corr=np.matmul(np.array([np.matmul(dtotal_dsum,w2)*(result1*(1-result1))]).T,[x[i]])
      w2=w2-learningrate*w2corr
      w1=w1-learningrate*np.delete(w1corr,0,0)
      if plot:
          nnplotter.ax.clear()
          nnplotter.plotweights(w1,0)
          nnplotter.plotweights(w2,1)
          nnplotter.plt.pause(plot_delay)
      if printy:
        print('y= ',expit(np.matmul(np.pad(expit(np.matmul(x,w1.T)),((0,0),(1,0)),'constant',constant_values=(1)),w2.T)))
  if printw:
    print('w1:\n',w1,'\nw2:\n',w2)
  return
