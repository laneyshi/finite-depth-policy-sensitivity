"""Log-odds state; interior motion updates align with the lattice."""
import numpy as np
from scipy.special import expit
LIMIT=86.4

def grid(bins):
    if bins==3:return np.linspace(-LIMIT,LIMIT,bins)
    # A refinement family: 1024 fine central intervals and 96 per tail per level.
    assert (bins-1)%38==0, bins
    level=(bins-1)/1216
    return np.r_[np.linspace(-LIMIT,-21.6,int(96*level)+1)[:-1],np.linspace(-21.6,21.6,int(1024*level)+1),np.linspace(21.6,LIMIT,int(96*level)+1)[1:]]
def project(e,bins):
    nodes=grid(bins);x=np.clip(e.logodds,-LIMIT,LIMIT)
    lo=np.clip(np.searchsorted(nodes,x,side="right")-1,0,bins-2)
    blend=(x-nodes[lo])/(nodes[lo+1]-nodes[lo])
    index=lo+(e.rng.random(e.n)<blend)
    e.logodds=grid(bins)[index];e.b=expit(e.logodds)
def initial_weights(bins):
    nodes=grid(bins);weights=np.zeros(bins)
    for i in range(bins-1):
        left=max(-np.log(9),nodes[i]);right=min(np.log(9),nodes[i+1])
        if right<=left:continue
        pl,pr=expit(left),expit(right);mass=pr-pl
        first=(right*pr-np.logaddexp(0.,right))-(left*pl-np.logaddexp(0.,left))
        upper=(first-nodes[i]*mass)/(nodes[i+1]-nodes[i])
        weights[i]+=(mass-upper)/.8;weights[i+1]+=upper/.8
    assert abs(weights.sum()-1)<1e-13
    return weights
