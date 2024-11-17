from numpy import array, matmul, pad, delete, sqrt , mean, tanh, add, arange, ndarray
from numpy import maximum as mx
from numpy.random import randn , seed
from scipy.special import expit
from aimaster.tools import nnplotter
from pickle import dump, load
from multiprocessing import Process, Queue, Pipe
from typing import List


class model:
    def __init__(self, architecture: List[int] = [], mtype: str = "sigmoid", staticinitialization: int = 0):
        """Creates a Neural Network model with given architecture and activation type.
        
        Args:
            architecture: List[int] with layer sizes [input neurons, hidden layer neurons, output neurons]
            mtype: str, one of "relu", "sigmoid", or "tanh"
            staticinitialization: int, seed for reproducibility
            
        Raises:
            ValueError: If architecture or mtype is invalid
        """
        # Validate model type
        valid_mtypes = {"relu", "sigmoid", "tanh"}
        if mtype not in valid_mtypes:
            raise ValueError(f"mtype must be one of {valid_mtypes}, got {mtype}")

        # Validate architecture
        if not isinstance(architecture, list):
            raise ValueError("architecture must be a list of integers")
        
        if architecture:
            if not all(isinstance(x, int) and x > 0 for x in architecture):
                raise ValueError("all elements in architecture must be positive integers")
            if len(architecture) < 2:
                raise ValueError("architecture must have at least 2 layers")

        if staticinitialization:
            if not isinstance(staticinitialization, int):
                raise ValueError("staticinitialization must be an integer")
            seed(staticinitialization)

        # Original initialization logic
        if not architecture:
            print("""architecture is not given as input.
                Assuming creating a dummy model to be used for loading a saved model...
                Initializing default neural network of architecture [2,2]. (2x2)...
                Initializing ...""")
            architecture = [2,2]

        self.architecture = architecture[:]
        architecture.append(0)
        
        # Initialize weights based on model type
        if mtype == "relu":
            self.W = array([randn(j,i+1)*sqrt(2/(i+1)) for i,j in zip(architecture[0::1],
                architecture[1::1])], dtype=object)[arange(len(architecture)-2)]
            self.currentmodeltype = "relu"
        elif mtype == "sigmoid":   
            self.W = array([randn(j,i+1)*sqrt(1/(i+1)) for i,j in zip(architecture[0::1],
                architecture[1::1])], dtype=object)[arange(len(architecture)-2)]
            self.currentmodeltype = "sigmoid"
        elif mtype == "tanh":   
            self.W = array([randn(j,i+1)*sqrt(1/(i+1)) for i,j in zip(architecture[0::1],
                architecture[1::1])], dtype=object)[arange(len(architecture)-2)]
            self.currentmodeltype = "tanh"

        print(f"\nModel initialized \n\n \t Architecture = {self.architecture} \n \t Model type = {self.currentmodeltype}\n\n Weights :")
        for i in range(len(self.W)):
            self.W[i][:,0] = 0  # Initializes bias weights to 0
            print('W[%d]=\n'%i, self.W[i], '\n')
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
    def predict(self, x: ndarray) -> ndarray:
        """Returns the network output for the given input.
        Note    if a single input is given to predict funcion it must be of shape
                (1,n) {n = no of input neurons}  
                Example: [1 , 1] has a shape of (2,) which is not accepted yet
                but [[1, 1]] has a shape of (1,2) which is desired if single input
                You know what you are doing :)
        Args:
            x: numpy.ndarray of shape (samples, features) matching input layer size
            
        Returns:
            numpy.ndarray of predictions
            
        Raises:
            ValueError: If input shape doesn't match network architecture
        """
        # Validate input
        try:
        # Attempt to convert the list to a numpy array if provided is not ndarray
            x = array(x)
        except TypeError as e:
            raise ValueError("Error: The input is not a valid list or array. {e}")
        if not isinstance(x, ndarray):
            raise ValueError("Input must be a numpy array")
            
        expected_features = self.architecture[0]
        if len(x.shape) != 2:
            raise ValueError(f"Input must be 2D array, got shape {x.shape}")
            
        if x.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} input features, got {x.shape[1]}")

        if self.currentmodeltype == "relu":
            return self.predictrelu(x)
        elif self.currentmodeltype == "sigmoid":
            return self.predictsigmoid(x)
        elif self.currentmodeltype == "tanh":
            return self.predicttanh(x)
        else:
            raise ValueError("Invalid model type")
    def savemodel(self, filename: str) -> None:
        """Saves the model in pickle format.
        
        Args:
            filename: str, path to save the model
            
        Raises:
            ValueError: If filename is invalid
        """
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        if not filename:
            raise ValueError("filename cannot be empty")
            
        with open(filename, 'wb') as file:
            dump(self, file)

    def loadmodel(filename: str) -> 'model':
        """Loads a model saved using aimaster modules.
         NOTE: learning rate is set default to 0.1 which sometimes is
        morethan required (result: gradient descent will not converge) or otherwise
        less than required (result: slow training). So feel free to experiment with 
        different values as this module is for basic understanding and experiments :)
        Adaptive learning rate will be introduced in future.
        Args:
            filename: str, path to the saved model
            
        Returns:
            model object
            
        Raises:
            ValueError: If filename is invalid
            FileNotFoundError: If file doesn't exist
        """
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        if not filename:
            raise ValueError("filename cannot be empty")
            
        try:
            with open(filename, 'rb') as file:
                return load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"No model file found at {filename}")
    def train(self, x: ndarray, y: ndarray, iterations: int, learningrate: float, 
              plot: bool = False, printy: bool = True, printw: bool = True,
              vmode: str = "queue", boost: float = 0, L2: float = 0) -> None:
        """Trains the neural network.
        
        Args:
            x: numpy.ndarray of training inputs
            y: numpy.ndarray of target outputs
            iterations: int, number of training iterations
            learningrate: float, learning rate
            plot: bool, whether to plot training progress
            printy: bool, whether to print predictions
            printw: bool, whether to print weights
            vmode: str, visualization mode ("queue" or "pipe")
            boost: float, boost parameter
            L2: float, L2 regularization parameter
            
        Raises:
            ValueError: If inputs are invalid or shapes don't match
        """
        # Validate inputs
        if not isinstance(x, ndarray) or not isinstance(y, ndarray):
            raise ValueError("x and y must be numpy arrays")
            
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise ValueError("x and y must be 2D arrays")
            
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have same number of samples, got {x.shape[0]} and {y.shape[0]}")
            
        if x.shape[1] != self.architecture[0]:
            raise ValueError(f"Expected {self.architecture[0]} input features, got {x.shape[1]}")
            
        if y.shape[1] != self.architecture[-1]:
            raise ValueError(f"Expected {self.architecture[-1]} output features, got {y.shape[1]}")
            
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
            
        if not isinstance(learningrate, (int, float)) or learningrate <= 0:
            raise ValueError("learningrate must be a positive number")
            
        if not isinstance(boost, (int, float)) or boost < 0:
            raise ValueError("boost must be a non-negative number")
            
        if not isinstance(L2, (int, float)) or L2 < 0:
            raise ValueError("L2 must be a non-negative number")
            
        if vmode not in {"queue", "pipe"}:
            raise ValueError("vmode must be 'queue' or 'pipe'")

        # Call appropriate training method
        if self.currentmodeltype == "sigmoid":
            self.trainsigmoid(x, y, iterations, learningrate, plot, printy, printw, vmode, boost, L2)
        elif self.currentmodeltype == "relu":
            self.trainrelu(x, y, iterations, learningrate, plot, printy, printw, vmode, boost, L2)
        elif self.currentmodeltype == "tanh":
            self.traintanh(x, y, iterations, learningrate, plot, printy, printw, vmode, boost, L2)
        else:
            raise ValueError("Invalid model type")
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
                        except Exception:
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
                        except Exception:
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
                        except Exception:
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
            except Exception:
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
            except Exception:
                cconn.close()
                cconn2.close()
                cconn3.close()
                nnplotter.plt.close()
                break