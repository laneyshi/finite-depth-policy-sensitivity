"""GPU-resident two-parameter jets with exact physical-state factorization.

No low-rank/Hessian approximation: only the two declared trainable parameters
are differentiated. Physical transitions are independent of the belief node.
"""
from pathlib import Path
import os,ctypes,threading
import numpy as np
import torch
from scipy.special import expit
from belief import grid,initial_weights,LIMIT
from exact_model import Model as CPUModel
from environment import MODE_PROBS,MOVES
from networks import load_endpoints
ROOT=Path(__file__).resolve().parent
MAP=[0,1,2,4,5,6,9,14,15]

def expand(a,fields):
    if fields==1:return a
    out=np.zeros((*a.shape[:-1],fields))
    if fields==4:out[...,:3]=a
    else:
        out[...,MAP]=a;out[...,8]=a[...,5]
    return out
class Jets:
    def __init__(self,value,fields):self.value=value;self.fields=fields
    def __getitem__(self,index):
        if isinstance(index,np.ndarray):index=torch.as_tensor(index,dtype=torch.long,device=self.value.device)
        return expand(self.value[index].cpu().numpy(),self.fields)
class Model:
    def __init__(self,bins=257,size=6,horizon=15,gamma=.9,**kwargs):
        self.bins,self.size,self.horizon,self.gamma=bins,size,horizon,gamma;self.n=size**6*bins
        self.backend=os.environ.get('SIM_COMPACT_BACKEND','cuda')
        self.device=torch.device(os.environ.get('SIM_DEVICE','cuda:0') if self.backend=='cuda' else 'cpu');self.compute_device=self.device;self.lock=threading.Lock()
        prior=os.environ.pop('SIM_FORWARD_DEVICE',None)
        try:base=CPUModel(bins=3,size=size,horizon=horizon,gamma=gamma)
        finally:
            if prior is not None:os.environ['SIM_FORWARD_DEVICE']=prior
        os.environ['SIM_MODEL_GPU_RESIDENT']='1' if self.backend=='cuda' else ''
        if self.compute_device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.9,self.compute_device)
        self.log_grid=grid(bins);self.grid=expit(self.log_grid)
        self.key=base.key[::3].copy();self.dec=base.dec[::3].copy();self.defender=base.defender[::3].copy()
        self.geometry=np.column_stack([self.defender,self.key-self.defender,self.dec-self.defender])/(size-1)
        phys_prob=base.initial.reshape(-1,3).sum(1);ph=np.flatnonzero(phys_prob);bw=initial_weights(bins);ib=np.flatnonzero(bw)
        self.initial_indices=(ph[:,None]*bins+ib).ravel()
        self.initial_weights=(phys_prob[ph,None]*bw[ib]).ravel()
        self.initial_weights/=self.initial_weights.sum()
        def tensor(a):return torch.as_tensor(np.ascontiguousarray(a),device=self.device)
        attacker_next=base.lower[::3]//3;delta=np.empty((size**6,16))
        for move in range(16):
            def likelihood(old,m):
                destinations=np.clip(old[:,None]+MOVES,0,size-1)
                return (destinations==destinations[:,m,None]).all(2)@MODE_PROBS.T
            lk=likelihood(self.key,move//4);lc=likelihood(self.dec,move%4)
            delta[:,move]=np.log(lk[:,1])+np.log(lc[:,0])-np.log(lk[:,0])-np.log(lc[:,1])
        self.terminal=base.terminal[::3].copy()
        self.compute_geometry=torch.as_tensor(self.geometry,device=self.compute_device)
        self.compute_belief=torch.as_tensor(self.grid,device=self.compute_device)
        self.compute_key_y=torch.as_tensor(self.key[:,1]/(size-1),device=self.compute_device)
        self.compute_terminal=torch.as_tensor(self.terminal.astype(bool),device=self.compute_device)
        self.params=[tensor(base.next[::3]//3),tensor(base.reward[::3]),tensor(base.timeout[::3]),tensor(base.alive[::3].astype(np.uint8)),
            tensor(base.p0[0][::3]),torch.empty((self.n,9),dtype=torch.float64,device=self.device),tensor(self.terminal),tensor(base.immediate_reward[::3]),
            tensor(attacker_next),tensor(delta),tensor(base.laws),tensor(self.grid),tensor(self.log_grid)]
        del base
        _,aware=load_endpoints(size);aw=[torch.as_tensor(w,device=self.compute_device) for w in aware]
        with torch.inference_mode():
            for start in range(0,self.n,65536):
                end=min(self.n,start+65536);ids=torch.arange(start,end,device=self.compute_device)
                tx=torch.cat([self.compute_geometry[ids//bins],self.compute_belief[ids%bins,None]],1)
                sx=torch.cat([tx[:,[0,1,4,5,2,3]],1-tx[:,6:7]],1)
                self.params[5][start:end]=(.5*(self.network(aw,tx).softmax(1)+self.network(aw,sx).softmax(1))).to(self.device)
        self.lib=ctypes.CDLL(str(ROOT/('bellman_compact.so' if self.backend=='cuda' else 'bellman_compact_cpu.so')));ptr=ctypes.c_void_p
        self.lib.select_device.argtypes=[ctypes.c_int];self.lib.select_device.restype=ctypes.c_int
        self.lib.step.argtypes=[ctypes.c_int]*4+[ptr]*14+[ctypes.c_double]*3+[ctypes.c_int]+[ptr]*2+[ctypes.c_size_t]
        self.lib.step.restype=ctypes.c_int
        print(f'{self.backend} compact model: {self.n:,} states, {bins} log-odds nodes, 2 trainable parameters',flush=True)
    @staticmethod
    def network(weights,x):
        for i in range(0,len(weights),2):
            x=x@weights[i].T+weights[i+1]
            if i<len(weights)-2:x=x.tanh()
        return x
    def policy(self,theta):
        logits=np.zeros((self.size,self.bins,4));logits[:,:,0]=theta[0]*(2*self.grid-1)
        logits[:,:,1]=theta[1]*(2*np.arange(self.size)[:,None]/(self.size-1)-1)
        return torch.as_tensor(logits,device=self.device).softmax(-1)
    def advance(self,value,mid,pi,kappa,t):
        if self.lib.select_device(self.device.index if self.device.type=='cuda' else 0):raise RuntimeError('Device selection failed')
        pointers=[ctypes.c_void_p(p.data_ptr()) for p in self.params]+[ctypes.c_void_p(pi.data_ptr())]
        err=self.lib.step(self.n,self.bins,self.size,value.shape[1],*pointers,float(kappa),float(self.gamma),LIMIT,int(t==self.horizon-1),
            ctypes.c_void_p(value.data_ptr()),ctypes.c_void_p(mid.data_ptr()),torch.cuda.current_stream(self.device).cuda_stream if self.device.type=='cuda' else 0)
        if err:raise RuntimeError(f'CUDA compact Bellman kernel error {err}')
    def evaluate(self,theta,kappa=.7,fields=17,depth=None,boundary=None,penalty=0.,rho=None,save_values=False):
        assert len(theta)==3 and theta[2]==0 and penalty==0 and rho is None
        with self.lock,torch.inference_mode():
            internal={1:1,4:3,17:9}[fields];depth=self.horizon if depth is None else depth;pi=self.policy(theta)
            value=torch.zeros((self.n,internal),dtype=torch.float64,device=self.device);mid=torch.empty_like(value)
            if boundary is not None:value[:,0]=torch.as_tensor(boundary,device=self.device)
            elif depth<self.horizon:
                b=torch.zeros((self.n,1),dtype=torch.float64,device=self.device);bm=torch.empty_like(b)
                for t in reversed(range(depth,self.horizon)):self.advance(b,bm,pi,kappa,t)
                value[:,0]=b[:,0];del b,bm
            values={depth:value[:,0].cpu().numpy().copy()} if save_values else None
            for t in reversed(range(depth)):
                self.advance(value,mid,pi,kappa,t)
                if save_values:values[t]=value[:,0].cpu().numpy().copy()
            jets=Jets(value,fields);r=self.initial_weights@jets[self.initial_indices]
            return r,jets,values
    def critic_boundary(self,weights,depth):
        with torch.inference_mode():
            value=torch.empty(self.n,dtype=torch.float64,device=self.device);ws=[torch.as_tensor(w,device=self.compute_device) for w in weights]
            for start in range(0,self.n,65536):
                end=min(self.n,start+65536);ids=torch.arange(start,end,device=self.compute_device);phys=ids//self.bins
                x=torch.cat([self.compute_geometry[phys],self.compute_belief[ids%self.bins,None],self.compute_key_y[phys,None],
                    torch.full((end-start,1),depth/self.horizon,dtype=torch.float64,device=self.compute_device)],1)
                pred=self.network(ws,x).ravel()*4*(self.size-1)
                value[start:end]=(pred*~self.compute_terminal[phys]).to(self.device)
            return value
