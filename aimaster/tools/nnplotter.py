import matplotlib.pyplot as plt
import numpy as np
def plotinit():
    global ax
    fig,ax=plt.subplots()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def plot(p,parameter,weight=1):
    global ax
    try :
        assert ax
    except Exception:
        print('plot not initialized')
        return
    if weight>0:
        ax.plot([parameter,parameter+1],[p[1],p[0]],linewidth=weight)
    elif weight<0:
        ax.plot([parameter,parameter+1],[p[1],p[0]],dashes=[4,2],linewidth=abs(weight))

def plotweights(w,parameter):
    for i in np.indices(w.shape).reshape(2,w.shape[0]*w.shape[1]).T:
        plot(i+(1,0),parameter,w[tuple(i)])