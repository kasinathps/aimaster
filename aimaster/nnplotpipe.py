import numpy as np
from numpy import array, matmul, pad, delete
from numpy import maximum as mx
from numpy.random import randn,rand
from scipy.special import expit
from aimaster.tools import nnplotter
from multiprocessing import Process, Queue, Pipe
from pickle import dump, load

class model:
    def __init__(self, architecture=[], type = "sigmoid"):
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
        if type=="relu":
            self.W=array([rand(j,i+1) for i,j in zip(architecture[0::1],
                architecture[1::1])])
        else:   
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

    def predictrelu(self,x):
        '''returns the network output of a specific input specifically
            NOTE:
                if a single input is given to predict funcion it must be of shape
                (1,n) {n = no of input neurons}  
                Example: [1 , 1] has a shape of (2,) which is not accepted yet
                but [[1, 1]] has a shape of (1,2) which is desired if single input
                You know what you are doing :) '''
        l=len(self.W)-1
        q=lambda z,y:mx(matmul(pad(x,((0,0),(1,0)),
          'constant',constant_values=1),self.W[z].T),0)if z==0 else (expit(matmul(pad(q(z-1,y),((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T)) if (z == y) else mx(matmul(pad(q(z-1,y),((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T),0))
        return q(l,l)

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

    def loadmodel(self,filename):
        with open(f"{filename}",'rb') as file:
            return load(file)
    def train(self,x,y,iterations,learningrate,plot=False,printy=True,printw=True,activation="sigmoid",):
        '''activation argument is used to select activation for neural network
        Specify Activation argument as < activation="sigmoid" > or "relu"
        Over the iterations, this function optimizes the values of all weights
        to reduce output error.
        if printy is set True (default) it prints the output on each iterations,
        if printw is set True (default) it prints the weight matrices on the 
        end of iterations 
        
        NOTE: learning rate is set default to 0.1 which sometimes is
        morethan required (result: gradient descent will not converge) or otherwise
        less than required (result: slow training). So feel free to experiment with 
        different values as this module is for basic understanding and experiments :)
        Adaptive learning rate will be introduced in future.
        '''
        if activation=="sigmoid":
            self.trainsigmoid(x,y,iterations,learningrate,plot,printy,printw)
        if activation=="relu":
            self.trainrelu(x,y,iterations,learningrate,plot,printy,printw)

    def trainsigmoid(self,x,y,iterations,learningrate,plot=False,printy=True,printw=True):
        '''Uses Sigmoid Activation.'''
        if plot:
            cconn,pconn = Pipe(duplex = False)
            pconn2,cconn2 = Pipe(duplex = False)
            cconn3,pconn3 = Pipe(duplex = False)
            Process(target=self.processplotter,args=(cconn,cconn2,cconn3,)).start()
        Wcorr=self.W*0
        lw= len(self.W)
        result=[[] for i in range(lw)]
        #Lsum=[[] for i in range(len(W))]
        p=lambda z:expit(matmul(pad(x,((0,0),(1,0)),
              'constant',constant_values=1),self.W[z].T))if z==0 else expit(matmul(
                  pad(p(z-1),((0,0),(1,0)),'constant',constant_values=1),self.W[z].T))
        try:
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
                if plot and pconn2.recv() == "Send":
                    Loss = (np.mean((self.predict(x)-y)**2))/len(x)
                    pconn.send(self.W)
                    pconn3.send([k,Loss])
                if printy:
                    print(self.predict(x))
                print('iteration : {}'.format(k+1))
        except KeyboardInterrupt:
            pass
        if printw:
          for i in range(lw):
              print('W[%d]=\n'%i,self.W[i],'\n')
        return
    def trainrelu(self,x,y,iterations,learningrate,plot=False,printy=True,printw=True):
        '''Relu Activation for Hidden layers and Sigmoid on final output.'''
        if plot:
            cconn,pconn = Pipe(duplex = False)
            pconn2,cconn2 = Pipe(duplex = False)
            cconn3,pconn3 = Pipe(duplex = False)
            Process(target=self.processplotter,args=(cconn,cconn2,cconn3,)).start()
        Wcorr=self.W*0
        lw= len(self.W)
        result=[[] for i in range(lw)]
        #Lsum=[[] for i in range(len(W))]
        if plot:
            nnplotter.plotinit()
        q=lambda z,y:mx(matmul(pad(x,((0,0),(1,0)),
          'constant',constant_values=1),self.W[z].T),0)if z==0 else (expit(matmul(pad(q(z-1,y),((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T)) if (z == y) else mx(matmul(pad(q(z-1,y),((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T),0))
        try:
            for k in range(iterations):
                for i in range(lw-1,-1,-1):
                    result[i]=pad(q(i,lw-1),((0,0),(1,0)),'constant',constant_values=1)
                for i in range(len(x)):
                    X=pad(x[i],((1,0)),'constant',constant_values=1)#input bias
                    for j in range(lw-1,-1,-1):
                        if j==lw-1:
                            Wcorr[j]=array([(result[j][i]-y[i])*(result[j][i]*(1-result[j][i]))])
                        else:
                            Wcorr[j]=(matmul(Wcorr[j+1][0][1:],self.W[j+1])*array([(result[j][i] >0)*1]))
                    for j in range(lw-1,-1,-1):
                        if j==0:
                            self.W[0]=self.W[0]-learningrate*delete(matmul(Wcorr[0].T,array([X])),0,0)
                        else:
                            self.W[j]=self.W[j]-learningrate*delete(matmul(Wcorr[j].T,array([result[j-1][i]])),0,0)
                if plot and pconn2.recv() == "Send":
                    Loss = (np.mean((self.predictrelu(x)-y)**2))/len(x)
                    pconn.send(self.W)
                    pconn3.send([k,Loss])
                if printy:
                    print(self.predictrelu(x))
                print('iteration : {}'.format(k+1))
        except KeyboardInterrupt:
            pass
        if printw:
          for i in range(lw):
              print('W[%d]=\n'%i,self.W[i],'\n')
        return
    def processplotter(self,cconn,cconn2,cconn3):
        nnplotter.plotinit()
        cconn2.send("Send")
        while True:
            nnplotter.ax.clear()
            try:
                tmp=cconn.recv()
                cconn2.send("")
                k=cconn3.recv()
            except:
                cconn.close()
                cconn2.close()
                cconn3.close()
                nnplotter.plt.close()
                break
            for i in range(len(tmp)):
                nnplotter.plotweights(tmp[i],i)
            nnplotter.ax.text(0,-0.25,s="iteration {: <5} Loss = {: <10}".format(str(k[0]),str(k[1])))
            nnplotter.plt.pause(0.00000001)
            try:
                cconn2.send("Send")
            except:
                cconn.close()
                cconn2.close()
                cconn3.close()
                nnplotter.plt.close()
                break
