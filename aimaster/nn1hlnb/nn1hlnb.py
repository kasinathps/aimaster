import numpy as np
from scipy.special import expit

def createnn(inputsize,hiddenlayersize,outputsize,pt=True):
  '''creates weight matrices w1 and w2 with appropriate weights
  according to inputsize,hiddenlayersize and outputsize
  if pt is set true (true by default) the function prints weight matrices 
  at the end , else if pt is set false it wont
  
  NOTE: this module doesnot include either input bias or hidden layer bias
  sot it wont meet specific examples which require some output on zero inputs
  eg: even a logic NOT gate'''
  global w1,w2
  w1=np.random.rand(hiddenlayersize,inputsize)
  w2=np.random.rand(outputsize,hiddenlayersize)
  if pt:
    print('w1:\n',w1,'\nw2:\n',w2)
  return

def weights():
  
  ''' prints out the weight matrices w1 and w2 '''
  
  global w1,w2
  print('w1:\n',w1,'\nw2:\n',w2)
  return

def predict(x):
  
  '''returns the network output of a specific input specifically
  NOTE: argument x must be of same format as used to train the 
  network'''
  
  global w1,w2
  return expit(np.matmul(expit(np.matmul(x,w1.T)),w2.T))

def train(X,y,iterations,learningrate=0.1,printy=True,printw=True):
  
  '''over the iterations, this function optimizes the values of w1 and
   w2 to reduce output error.
  if printy is set True (default) it prints the output on each iterations,
  if printw is set True (default) it prints the weight matrices on the 
  end of iterations 
  
  NOTE: learning rate is set default to 0.1 which sometimes is
  morethan required (result: gradient descent will not converge) or less 
  than required (result: slow training). feel free to experiment with 
  different values as this module is for basic understanding :) '''
  global w1,w2
  for j in range(iterations):
    for i in range(len(X)):
      Hsum=np.matmul(X[i],w1.T)
      result1=expit(Hsum)
      sum=np.matmul(result1,w2.T)
      result=expit(sum)
      e0=result-y[i]
      dtotal_dsum=e0*result*(1-result)
      w2corr=np.matmul(dtotal_dsum.reshape(-1,1),[result1])
      w1corr=np.matmul(np.array([np.matmul(dtotal_dsum,w2)*(result1*(1-result1))]).T,[X[i]])
      w2=w2-learningrate*w2corr
      w1=w1-learningrate*w1corr
      if printy:
        print('y= ',expit(np.matmul(expit(np.matmul(X,w1.T)),w2.T)))
  if printw:
    print('w1:\n',w1,'\nw2:\n',w2)
  return
