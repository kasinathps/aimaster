import numpy as np
from numpy import array, matmul, pad, delete, sqrt
from numpy import maximum as mx
from numpy.random import randn,rand
from scipy.special import expit
from aimaster.tools import nnplotter
from multiprocessing import Process, Queue, Pipe
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
        self.W=array([randn(j,i+1)*np.sqrt(2/(i+1)) for i,j in zip(architecture[0::1],
            architecture[1::1])])
        for i in range(len(self.W)):
            self.W[i][:,0]=0
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
        l=len(self.W)-1
        q=lambda z,y:mx(matmul(pad(x,((0,0),(1,0)),
          'constant',constant_values=1),self.W[z].T),0)if z==0 else (expit(matmul(pad(q(z-1,y),((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T)) if (z == y) else mx(matmul(pad(q(z-1,y),((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T),0))
        return q(l,l)

    def savemodel(self,filename):
        """Saves the model in pickle format"""
        with open(f"{filename}",'wb') as file:
            dump(self,file)

    def loadmodel(self,filename):
        with open(f"{filename}",'rb') as file:
            return load(file)
    def train(self,x,y,iterations,learningrate,plot=False,printy=True,printw=True,vmode="queue"):
        '''Relu Activation for Hidden layers and Sigmoid on final output.'''
        if plot:
            if vmode=="queue":
                event_q = Queue()
                send_q = Queue()
                p = Process(target=self.processplotterqueue,args=(event_q,send_q,))
                p.start()
                send_q.get(block=True , timeout=3)#checking startsignal
            elif vmode=="pipe":
                cconn,pconn = Pipe(duplex = False)
                pconn2,cconn2 = Pipe(duplex = False)
                cconn3,pconn3 = Pipe(duplex = False)
                Process(target=self.processplotterpipe,args=(cconn,cconn2,cconn3,)).start()
            else:
                print("visualization mode unknown. Turning off plotting")
                plot=False
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
                Loss = (np.mean((self.predict(x)-y)**2))
                if plot:
                    if vmode == "queue":
                        try:
                            if send_q.get_nowait()=="Send" and k!=iterations-1:
                                event_q.put([self.W,k,Loss])
                        except:
                            pass
                    else:
                        if pconn2.recv() == "Send":
                            pconn.send(self.W)
                            pconn3.send([k,Loss])
                if printy:
                    print(str(self.predict(x))+'\n iteration :'+ str(k+1))
                else:
                    print('iteration : {}'.format(k+1))
        except KeyboardInterrupt:
            pass
        if printw:
          for i in range(lw):
              print('W[%d]=\n'%i,self.W[i],'\n')
        if plot and vmode == "queue":
            event_q.put("close")
            nnplotter.plt.close()
            p.join()
        return

    def processplotterqueue(self,event_q,send_q):
        nnplotter.plotinit()
        send_q.put("Startsignal")
        send_q.put("Send")
        while True:
            nnplotter.ax.clear()
            tmp=event_q.get(block=True)
            if type(tmp)==str and tmp == "close":
                event_q.close()
                send_q.close()
                nnplotter.plt.close()
                break
            else:
                #print("Queue Received")#for debugging purposes
                for i in range(len(tmp[0])):
                    nnplotter.plotweights(tmp[0][i],i)
                nnplotter.ax.text(0,-0.25,s="iteration {: <5} Loss = {: <10}".format(tmp[1],tmp[2]))
                nnplotter.plt.pause(0.00000001)
                send_q.put("Send")
    def processplotterpipe(self,cconn,cconn2,cconn3):
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
