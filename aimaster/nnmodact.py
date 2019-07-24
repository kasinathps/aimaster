import numpy as np
from numpy import array, matmul, pad, delete
from numpy.random import randn
from scipy.special import expit
from aimaster.tools import nnplotter
from pickle import dump, load

class model:
    def __init__(self, architecture=[]):
        if not architecture :
            print("""architecture is not given as input.
                Assuming creating a dummy model to be used for loading a saved model...
                Initializing default neural network of architecture [2,2]. (2x2)...
                Initializing ...""")
            architecture = [2,2]
        if not isinstance(architecture,list):
            print("""give architecture as a list of neurons in each layer including
                input layer,hidden layers and output layer!
                Example: architecture=[2,3,3,1] for nn with input layer of 
                2 neurons, two hidden layers with 3 neurons each(without 
                counting bias) and an output layer of 1 neuron""")
            return None
        self.arch=architecture
        self.W=array([randn(j,i+1) for i,j in zip(architecture[0::1],
            architecture[1::1])])
        for i in range(len(self.W)):
            print('W[%d]=\n'%i,self.W[i],'\n')
        return

    def weights(self,plot=False):
        '''prints out the weight matrixes w1 and w2'''
        if plot:
            nnplotter.plotinit()
            for i in range(len(self.W)):
                nnplotter.plotweights(self.W[i],i)
            nnplotter.plt.show()
        for i in range(len(self.W)):
            print('W[%d]=\n'%i,self.W[i],'\n')
        return

    def predict(self,x):
        '''returns the network output of a specific input specifically
            NOTE:
                if a single input is given to predict funcion it must be of shape
                (1,n) {n = no of input neurons}  
                Example: [1 , 1] has a shape of (2,) which is not accepted yet
                but [[1, 1]] has a shape of (1,2) which is desired if single input
                You know what you are doing :) '''
        p=lambda z:expit(matmul(pad(x,((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T))if z==0 else expit(matmul(
                pad(p(z-1),((0,0),(1,0)),
                       'constant',constant_values=1),self.W[z].T))
        return p(len(self.W)-1)

    def savemodel(self,filename):
        """Saves the model in pickle format"""
        with open(f"{filename}",'wb') as file:
            dump(self,file)

    def loadmodel(filename):
        with open(f"{filename}",'rb') as file:
            return load(file)

    def trainsigmoid(self,x,y,iterations,learningrate=0.1,plot=False,Plotfreq=1,Plotmaxm=0,printy=True,printw=True,plot_delay=0.00000001):
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
      Wcorr=self.W*0
      lw= len(self.W)
      result=[[] for i in range(lw)]
      #Lsum=[[] for i in range(len(W))]
      if plot:
          nnplotter.plotinit()
      p=lambda z:expit(matmul(pad(x,((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T))if z==0 else expit(matmul(
                pad(p(z-1),((0,0),(1,0)),'constant',constant_values=1),self.W[z].T))
      for k in range(iterations):
          for i in range(lw-1,-1,-1):
              result[i]=pad(p(i),((0,0),(1,0)),'constant',constant_values=1)
          for i in range(len(x)):
              X=pad(x[i],((1,0)),'constant',constant_values=1)
              for j in range(lw-1,-1,-1):
                  if j==lw-1:
                      Wcorr[j]=array([(result[j][i]-y[i])*(result[j][i]*(1-result[j][i]))])#(pred - expected)*(derivative of activation)
                  else:
                      Wcorr[j]=(matmul(Wcorr[j+1][0][1:],self.W[j+1])*array([(result[j][i]*(1-result[j][i]))]))
              for j in range(lw-1,-1,-1):
                  if j==0:
                      self.W[0]=self.W[0]-learningrate*delete(matmul(Wcorr[0].T,array([X])),0,0)
                  else:
                      self.W[j]=self.W[j]-learningrate*delete(matmul(Wcorr[j].T,array([result[j-1][i]])),0,0)
          if plot:
              if k==0 and Plotmaxm:
                  figManager = nnplotter.plt.get_current_fig_manager()
                  figManager.window.showMaximized()
              if k%Plotfreq == 0:
                  nnplotter.ax.clear()
                  for i in range(lw):
                      nnplotter.plotweights(self.W[i],i)
                  nnplotter.ax.text(0,0,s='iteration {}'.format(k))
                  nnplotter.plt.pause(plot_delay)
          if printy:
              print(self.predict(x))
          print('iteration : {}'.format(k+1))
      if printw:
        for i in range(lw):
            print('W[%d]=\n'%i,self.W[i],'\n')
      return

