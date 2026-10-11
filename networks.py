from pathlib import Path
import os
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent
CIRCLE_RADII=np.array([float(z) for z in os.environ.get('DESIGNED_CIRCLE_RADII','3,6').split(',')])

class Network(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64),
                                 nn.Tanh(), nn.Linear(64, outputs))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2)); nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, .01 if outputs > 1 else 1.)

    def forward(self, x): return self.net(x)

def numpy_net(net):
    if isinstance(net, SharedActor):
        return {'shared':net.theta.detach().numpy().astype(float)}
    if isinstance(net, CircleActor):
        return {'circle': net.theta.detach().numpy().astype(float),'radii':CIRCLE_RADII.copy()}
    if isinstance(net, StructuredActor):
        return {'structured': net.theta.detach().numpy().astype(float)}
    return [p.detach().cpu().numpy().astype(np.float64) for p in net.parameters()]

def forward(weights, x):
    if isinstance(weights, dict):
        if 'shared' in weights:
            return np.tile(np.array([-weights['shared'][0],weights['shared'][0],0.,0.]),(len(x),1))
        f=2*x[:,[6,7]]-1
        if 'circle' in weights:
            angle=float(weights['circle'][0]);r=weights.get('radii',CIRCLE_RADII)
            coeff=np.array([r[0]*np.cos(angle),r[1]*np.sin(angle)])
        else:coeff=weights['structured']
        return np.column_stack([f*coeff,np.zeros((len(x),2))])
    if os.environ.get('SIM_FORWARD_DEVICE'):
        from gpu_forward import forward as accelerated_forward
        return accelerated_forward(weights,x)
    for i in range(0, len(weights), 2):
        x = x @ weights[i].T + weights[i+1]
        if i < len(weights)-2: x = np.tanh(x)
    return x

class StructuredActor(nn.Module):
    """Declared two-parameter policy family; NOT a full-network optimum claim."""
    def __init__(self):
        super().__init__()
        self.theta=nn.Parameter(torch.zeros(2))

    def forward(self,x):
        return torch.cat([(2*x[:,[6,7]]-1)*self.theta,torch.zeros_like(x[:,:2])],1)

class CircleActor(nn.Module):
    """Fixed-norm strategic coefficients, smoothly parametrized by angle.

    An explicitly restricted family with finite parameters and bounded logits.
    No penalty or entropy is added to the evaluated return objective.
    """
    def __init__(self):
        super().__init__();self.theta=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        coeff=torch.cat([CIRCLE_RADII[0]*self.theta.cos(),CIRCLE_RADII[1]*self.theta.sin()])
        return torch.cat([(2*x[:,[6,7]]-1)*coeff,torch.zeros_like(x[:,:2])],1)

class SharedActor(nn.Module):
    """Scalar identifiable softmax family: logits (-theta,theta,0,0)."""
    def __init__(self):
        super().__init__();self.theta=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return torch.cat([-self.theta,self.theta,self.theta*0,self.theta*0]).expand(len(x),4)

def softmax(z):
    p = np.exp(z-z.max(1, keepdims=True)); return p/p.sum(1, keepdims=True)

def endpoints(e, blind, aware):
    # Symmetrization removes accidental label bias and permits exact key-order
    # reduction in the separate small finite-state benchmark.
    live=~e.done
    result0=np.full((e.n,9),1/9);result1=result0.copy()
    if not live.any():return result0,result1
    x0, x1 = e.observations('blind')[live], e.observations('aware')[live]
    s0 = x0[:, [0,1,4,5,2,3]]
    s1 = np.column_stack([s0, 1-e.b[live]])
    p0 = .5*(softmax(forward(blind,x0))+softmax(forward(blind,s0)))
    p1 = .5*(softmax(forward(aware,x1))+softmax(forward(aware,s1)))
    result0[live]=p0;result1[live]=p1
    return result0,result1

def load_endpoints(size):
    path = ROOT/f'results/endpoints_g{size}.pt'
    if not path.exists():
        raise FileNotFoundError(f'{path.name} is missing. Run python run.py endpoints first.')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    b, a = Network(6,9), Network(7,9)
    b.load_state_dict(ck['blind']); a.load_state_dict(ck['aware'])
    return numpy_net(b), numpy_net(a)
