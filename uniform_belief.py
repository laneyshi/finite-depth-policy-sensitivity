"""Log-odds state; interior motion updates align with the lattice."""
import numpy as np
from scipy.special import expit
LIMIT=86.4

def grid(bins):return np.linspace(-LIMIT,LIMIT,bins)
def project(e,bins):
    scaled=np.clip((e.logodds+LIMIT)*(bins-1)/(2*LIMIT),0,bins-1)
    lo=np.minimum(scaled.astype(int),bins-2)
    index=lo+(e.rng.random(e.n)<scaled-lo)
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
