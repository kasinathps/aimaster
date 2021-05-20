from numpy import array, matmul, pad, delete, sqrt , mean, tanh, add, arange
from numpy import maximum as mx
from numpy.random import randn , seed
from scipy.special import expit
from aimaster.tools import nnplotter
from pickle import dump, load
from multiprocessing import Process, Queue, Pipe

class model:
    def __init__(self, architecture=[], mtype = "sigmoid",staticinitialization=0):
        """ Creates a Neural Network model with given architecture and activation type.
            architecture = [input neurons,hidden layer neurons, output neurons]
                Example :
                    m1 = model([2,3,3,1]) gives nn with 2 inputs plus a bias,
                    two hidden layers with 3 neurons each with additional bias,
                    and connected to one output neuron
                    Bias Weights are initialized to ZERO.
            mtype : modeltype   "relu", "sigmoid" and "tanh" are supported.
            staticinitialization :   is fed into numpy.random.seed() to generate same 
                model parameters over different machines or different times"""
        if staticinitialization:
            seed(staticinitialization)
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
            return -1
        self.architecture=architecture[:]
        architecture.append(0)
        if mtype=="relu":
            self.W=array([randn(j,i+1)*sqrt(2/(i+1)) for i,j in zip(architecture[0::1],
                architecture[1::1])])[arange(len(architecture)-2)]
            self.currentmodeltype="relu"
        elif mtype == "sigmoid":   
            self.W=array([randn(j,i+1)*sqrt(1/(i+1)) for i,j in zip(architecture[0::1],
                architecture[1::1])])[arange(len(architecture)-2)]
            self.currentmodeltype="sigmoid"
        elif mtype == "tanh":   
            self.W=array([randn(j,i+1)*sqrt(1/(i+1)) for i,j in zip(architecture[0::1],
                architecture[1::1])])[arange(len(architecture)-2)]
            self.currentmodeltype="tanh"
        else:
            print("model type is unknown or not set properly")
            return -1
        print(f"\nModel initialized \n\n \t Architecture = {self.architecture} \n \t Model type = {self.currentmodeltype}\n\n Weights :")
        for i in range(len(self.W)):
            self.W[i][:,0]=0#Initializes bias weights to 0
            print('W[%d]=\n'%i,self.W[i],'\n')
        return None
    def weights(self,plot=False):
        """prints out the weight matrixes w1 and w2"""
        for i in range(len(self.W)):
            print('W[%d]=\n'%i,self.W[i],'\n')
        if plot:
            nnplotter.plotinit()
            for i in range(len(self.W)):
                nnplotter.plotweights(self.W[i],i)
            nnplotter.plt.show()
        return 0
    def predictrelu(self,x):
        """returns the network output of a specific input specifically using relu activation"""
        l=len(self.W)-1
        q=lambda z,y:mx(matmul(pad(x,((0,0),(1,0)),
          'constant',constant_values=1),self.W[z].T),0)if z==0 else (expit(matmul(pad(q(z-1,y),((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T)) if (z == y) else mx(matmul(pad(q(z-1,y),((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T),0))
        return q(l,l)
    def predictsigmoid(self,x):
        """returns the network output of a specific input specifically using sigmoid activation"""
        p=lambda z:expit(matmul(pad(x,((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T))if z==0 else expit(matmul(
                pad(p(z-1),((0,0),(1,0)),
                       'constant',constant_values=1),self.W[z].T))
        return p(len(self.W)-1)
    def predicttanh(self,x):
        """returns the network output of a specific input specifically using tanh activation"""
        p=lambda z:tanh(matmul(pad(x,((0,0),(1,0)),
            'constant',constant_values=1),self.W[z].T))if z==0 else tanh(matmul(
                pad(p(z-1),((0,0),(1,0)),
                       'constant',constant_values=1),self.W[z].T))
        return p(len(self.W)-1)
    def predict(self,x ):
        """returns the network output of a specific input specifically
            NOTE:
                if a single input is given to predict funcion it must be of shape
                (1,n) {n = no of input neurons}  
                Example: [1 , 1] has a shape of (2,) which is not accepted yet
                but [[1, 1]] has a shape of (1,2) which is desired if single input
                You know what you are doing :) """
        if self.currentmodeltype == "relu":
            return(self.predictrelu(x))
        elif self.currentmodeltype == "sigmoid":
            return(self.predictsigmoid(x))
        elif self.currentmodeltype == "tanh":
            return(self.predicttanh(x))
        else:
            print("currentmodeltype is unknown or not set properly")
            return None
    def savemodel(self,filename):
        """Saves the model in pickle format"""
        with open(f"{filename}",'wb') as file:
            dump(self,file)

    def loadmodel(self,filename):
        """Loads a model saved using aimaster modules"""
        with open(f"{filename}",'rb') as file:
            return load(file)
    def train(self,x,y,iterations,learningrate,plot=False,printy=True,printw=True,vmode="queue",boost=0,L2=0):
        """activation argument is used to select activation for neural network
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
        """
        if self.currentmodeltype=="sigmoid":
            self.trainsigmoid(x,y,iterations,learningrate,plot,printy,printw,vmode,boost,L2)
        elif self.currentmodeltype=="relu":
            self.trainrelu(x,y,iterations,learningrate,plot,printy,printw,vmode,boost,L2)
        elif self.currentmodeltype=="tanh":
            self.traintanh(x,y,iterations,learningrate,plot,printy,printw,vmode,boost,L2)
        else:
            print("Either currentmodeltype not set or corrupt. Check again")
            return None
    def trainsigmoid(self,x,y,iterations,learningrate,plot=False,printy=True,printw=True,vmode="queue",boost=0,L2=0):
        """Uses Sigmoid Activation."""
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
                Loss = (mean((self.predictsigmoid(x)-y)**2))
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
                    print(str(self.predictsigmoid(x))+'\n iteration :'+ str(k+1))
                else:
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
    def traintanh(self,x,y,iterations,learningrate,plot=False,printy=True,printw=True,vmode="queue",boost=0,L2=0):
        """Uses tanh Activation."""
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
        result=[[] for i in range(lw)]
        #Lsum=[[] for i in range(len(W))]
        p=lambda z:tanh(matmul(pad(x,((0,0),(1,0)),
              'constant',constant_values=1),self.W[z].T))if z==0 else tanh(matmul(
                  pad(p(z-1),((0,0),(1,0)),'constant',constant_values=1),self.W[z].T))
        try:
            for k in range(iterations):
                for i in range(lw-1,-1,-1):
                    result[i]=pad(p(i),((0,0),(1,0)),'constant',constant_values=1)
                for i in range(len(x)):
                    X=pad(x[i],((1,0)),'constant',constant_values=1)
                    for j in range(lw-1,-1,-1):
                        if j==lw-1:
                            Wcorr[j]=array([(result[j][i]-y[i])*(1-(result[j][i])**2)])#(pred - expected)*(derivative of activation)
                        else:
                            Wcorr[j]=(matmul(Wcorr[j+1][0][1:],self.W[j+1])*array([(1-(result[j][i])**2)]))
                    for j in range(lw-1,-1,-1):
                        if j==0:
                            self.W[0]=self.W[0]-learningrate*delete(matmul(Wcorr[0].T,array([X])),0,0)
                        else:
                            self.W[j]=self.W[j]-learningrate*delete(matmul(Wcorr[j].T,array([result[j-1][i]])),0,0)
                Loss = (mean((self.predicttanh(x)-y)**2))
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
                    print(str(self.predicttanh(x))+'\n iteration :'+ str(k+1))
                else:
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
    def trainrelu(self,x,y,iterations,learningrate,plot=False,printy=True,printw=True,vmode="queue",boost=0,L2=0):
        """Relu Activation for Hidden layers and Sigmoid on final output."""
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
                Loss = (mean((self.predictrelu(x)-y)**2))
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
                    print(str(self.predictrelu(x))+'\n iteration :'+ str(k+1))
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
                #print("Queue Received")#for debugging purposes
                for i in range(len(tmp[0])):
                    nnplotter.plotweights(tmp[0][i],i)
                nnplotter.ax.text(0,-0.25,s="iteration {: <5} Loss = {: <10}".format(tmp[1],tmp[2]))
                nnplotter.plt.pause(0.001)
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
            nnplotter.plt.pause(0.001)
            try:
                cconn2.send("Send")
            except:
                cconn.close()
                cconn2.close()
                cconn3.close()
                nnplotter.plt.close()
                break
