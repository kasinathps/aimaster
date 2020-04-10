import numpy as np
from numpy import array, matmul, pad, delete, sqrt, mean, add
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
        self.W=array([randn(j,i+1)*np.sqrt(1/(i+1)) for i,j in zip(architecture[0::1],
            architecture[1::1])])
        for i in range(len(self.W)):
            self.W[i][:,0]=0
            print('W[%d]=\n'%i,self.W[i],'\n')
        return
    def weights(self,plot=False):
        """prints out the weight matrixes w1 and w2"""
        for i in range(len(self.W)):
            print('W[%d]=\n'%i,self.W[i],'\n')
        if plot:
            nnplotter.plotinit()
            for i in range(len(self.W)):
                nnplotter.plotweights(self.W[i],i)
            nnplotter.plt.show()
        return
    def predict(self,x):
        """returns the network output of a specific input specifically
            NOTE:
                if a single input is given to predict funcion it must be of shape
                (1,n) {n = no of input neurons}  
                Example: [1 , 1] has a shape of (2,) which is not accepted yet
                but [[1, 1]] has a shape of (1,2) which is desired if single input
                You know what you are doing :) """
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
    def train(self,x,y,iterations,learningrate,plot=False,printy=True,printw=True,vmode="queue",boost=0,L2=0):
        """Uses Sigmoid Activation. Has asynchronous and synchronous plotting feature, L1 regularization 
        and experimental boost feature."""
        if plot:
            if vmode=="queue":
                event_q = Queue()
                send_q = Queue()
                q = Process(target=self.processplotterqueue,args=(event_q,send_q,))
                q.start()
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
        lx=len(x)
        if boost:
            tmp=Wcorr
            boostcounter = abs(y-self.predict(x)).argmax()
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
                    if boost:
                        boostcounter = abs(y-self.predict(x)).argmax()
                    X=pad(x[i],((1,0)),'constant',constant_values=1)
                    for j in range(lw-1,-1,-1):
                        if j==lw-1:
                            Wcorr[j]=array([(result[j][i]-y[i])*(result[j][i]*(1-result[j][i]))])#(pred - expected)*(derivative of activation)
                        else:
                            Wcorr[j]=(matmul(Wcorr[j+1][0][1:],self.W[j+1])*array([(result[j][i]*(1-result[j][i]))]))
                    if boost:
                        tmp=tmp+Wcorr
                        if i == boostcounter:
                            tmp=Wcorr
                        if(any([abs(subw.max()) > boost for subw in tmp])):
                            tmp = Wcorr
                    for j in range(lw-1,-1,-1):
                        if j==0:
                            self.W[0]=self.W[0]-learningrate*(delete(matmul(Wcorr[0].T,array([X])),0,0)+((L2/2*lx)*self.W[0]))
                        else:
                            self.W[j]=self.W[j]-learningrate*(delete(matmul(Wcorr[j].T,array([result[j-1][i]])),0,0)+((L2/2*lx)*self.W[j]))
                    #following loop lines currently boost the sigmoid(experimental).
                    if boost :
                        for j in range(lw-1,-1,-1):
                            if j==0:
                                self.W[0]=self.W[0]-learningrate*(delete(matmul(tmp[0].T,array([X])),0,0)+((L2/2*lx)*self.W[0]))
                            else:
                                self.W[j]=self.W[j]-learningrate*(delete(matmul(tmp[j].T,array([result[j-1][i]])),0,0)+((L2/2*lx)*self.W[j]))
                Loss = (mean((self.predict(x)-y)**2))/len(x)
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
                    print(self.predict(x))
                print('iteration : {}'.format(k+1))
        except KeyboardInterrupt:
            pass
        if printw:
          for i in range(lw):
              print('W[%d]=\n'%i,self.W[i],'\n')
        if plot and vmode == "queue":
            event_q.put("close")
            q.join()
        return 0
    def trainminimal(self,x,y,iterations,learningrate,L2=0):
        """Uses Sigmoid Activation.Has no plotting Feature.But have l1 regularization"""
        Wcorr=self.W*0
        lw= len(self.W)
        lx=len(x)
        result=[[] for i in range(lw)]
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
                            Wcorr[j]=array([(result[j][i]-y[i])*(result[j][i]*(1-result[j][i]))])
                        else:
                            Wcorr[j]=(matmul(Wcorr[j+1][0][1:],self.W[j+1])*array([(result[j][i]*(1-result[j][i]))]))
                    for j in range(lw-1,-1,-1):
                        if j==0:
                            self.W[0]=self.W[0]-learningrate*(delete(matmul(Wcorr[0].T,array([X])),0,0)+((L2/2*lx)*self.W[0]))
                        else:
                            self.W[j]=self.W[j]-learningrate*(delete(matmul(Wcorr[j].T,array([result[j-1][i]])),0,0)+((L2/2*lx)*self.W[j]))
                Loss = (mean((self.predict(x)-y)**2))
                print('Loss'+str(Loss)+'\t iteration : {}'.format(k+1))
        except KeyboardInterrupt:
            pass
        for i in range(lw):
            print('W[%d]=\n'%i,self.W[i],'\n')
        print(self.predict(x))
        return 0
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
