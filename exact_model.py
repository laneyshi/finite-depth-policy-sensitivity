"""Finite-belief approximation with exact conditional Bellman integration.

Key-ordered coordinates quotient out attacker-label permutation symmetry.
The defender policy is explicitly symmetrized, so this is an exact symmetry
reduction. Belief interpolation is a disclosed stochastic grid approximation;
its resolution is checked separately. Rewards/dynamics follow the declared designed_validation environment.
"""
from pathlib import Path
import ctypes
import sys
import time
import os
import numpy as np
import torch
from scipy.special import expit
from belief import grid, initial_weights, LIMIT

from environment import MODE_PROBS, MOVES, DEFENDER_MOVES, DIVERSION_WEIGHT, starting_row, SMALL_CAPTURE_RADIUS
from networks import load_endpoints, forward, softmax

ROOT=Path(__file__).resolve().parent


def features(a0, a1, defender, belief, size):
    # A narrow, identifiable policy family, with signed, state-dependent logits.
    distance0=np.abs(a0-defender).sum(1)
    distance1=np.abs(a1-defender).sum(1)
    return np.column_stack([2*belief-1, 2*a0[:,1]/(size-1)-1,
                             (distance1-distance0)/(2*(size-1))])


def probabilities(theta, feature):
    logits=np.column_stack([feature*np.asarray(theta)[None],np.zeros(len(feature))])
    logits-=logits.max(1,keepdims=True)
    p=np.exp(logits);return p/p.sum(1,keepdims=True)


def symmetric_endpoints(a0,a1,defender,belief,t,horizon,size,blind,aware):
    x=np.column_stack([defender,a0-defender,a1-defender])/(size-1)
    y=x[:,[0,1,4,5,2,3]]
    p0=.5*(softmax(forward(blind,x))+softmax(forward(blind,y)))
    p1=.5*(softmax(forward(aware,np.column_stack([x,belief])))+
           softmax(forward(aware,np.column_stack([y,1-belief]))))
    return p0,p1


class Model:
    def __init__(self,bins=9,size=6,horizon=15,gamma=.9,checkpoint=None,cache=None):
        self.size,self.horizon,self.gamma,self.bins=size,horizon,gamma,bins
        # Uniform probability grid includes endpoints; interpolate exact Bayes posterior.
        self.log_grid=grid(bins);self.grid=expit(self.log_grid)
        self.n=size**6*bins
        self.lib=ctypes.CDLL(str(ROOT/'bellman.so'))
        ptr=ctypes.c_void_p
        self.lib.defender_step.argtypes=[ctypes.c_int,ctypes.c_int]+[ptr]*8+[ctypes.c_double,ptr]
        self.lib.attacker_step.argtypes=[ctypes.c_int,ctypes.c_int]+[ptr]*8
        self.lib.defender_step.restype=None;self.lib.attacker_step.restype=None
        self.cache=Path(cache or ROOT/'cache')/f'g{size}_b{bins}'
        self.cache.mkdir(parents=True,exist_ok=True)
        npos=size*size
        idx=np.arange(self.n,dtype=np.int64)
        self.ib=(idx%bins).astype(np.int32);idx//=bins
        self.defpos=(idx%npos).astype(np.int32);idx//=npos
        self.decpos=(idx%npos).astype(np.int32);idx//=npos
        self.keypos=idx.astype(np.int32)
        self.xy=lambda p:np.column_stack([p%size,p//size])
        self.key,self.dec,self.defender=map(self.xy,(self.keypos,self.decpos,self.defpos))
        self.belief=self.grid[self.ib]
        self.logodds=self.log_grid[self.ib]
        self.feature=np.ascontiguousarray(features(self.key,self.dec,self.defender,self.belief,size))
        self.terminal=np.ascontiguousarray(((self.keypos==self.defpos)|
                          (self.decpos==self.defpos)|(self.key[:,1]==0)).astype(np.uint8))
        self.initial=self.initial_distribution()
        self.laws=np.empty((4,16))
        for a in range(4):
            self.laws[a]=np.outer(MODE_PROBS[a//2],MODE_PROBS[a%2]).ravel()
        self.build()
        blind,aware=load_endpoints(size)
        # The blind endpoint repeats identically at all belief nodes.
        x=np.column_stack([self.defender[::bins],self.key[::bins]-self.defender[::bins],
                           self.dec[::bins]-self.defender[::bins]])/(size-1)
        y=x[:,[0,1,4,5,2,3]]
        physical_p0=.5*(softmax(forward(blind,x))+softmax(forward(blind,y)))
        p0=np.repeat(physical_p0,bins,axis=0);p1=np.empty_like(p0)
        # NumPy's installed OpenBLAS exposes a process-local thread control.
        # Restore the prior setting before parallel depth evaluations.
        libs=list((Path(np.__file__).resolve().parent.parent/'numpy.libs').glob('*openblas*.so'))
        setter=None;prior=1
        if libs:
            lib=ctypes.CDLL(str(libs[0]))
            setter=getattr(lib,'scipy_openblas_set_num_threads64_',None)
            getter=getattr(lib,'scipy_openblas_get_num_threads64_',None)
            if setter is not None and getter is not None:
                setter.argtypes=[ctypes.c_int];prior=getter();setter(8)
            else:setter=None
        try:
            for start in range(0,self.n,65536):
                sl=slice(start,start+65536)
                x=np.column_stack([self.defender[sl],self.key[sl]-self.defender[sl],
                                   self.dec[sl]-self.defender[sl]])/(size-1)
                y=x[:,[0,1,4,5,2,3]]
                p1[sl]=.5*(softmax(forward(aware,np.column_stack([x,self.belief[sl]])))+
                           softmax(forward(aware,np.column_stack([y,1-self.belief[sl]]))))
        finally:
            if setter is not None:setter(prior)
        self.p0=[p0]*horizon;self.p1=[p1]*horizon
        print(f'model: {self.n:,} states, {bins} belief nodes, conditional sums exact',flush=True)

    def pack(self,k,c,d,b):
        return (((k.astype(np.int64)*self.size**2+c)*self.size**2+d)*self.bins+b).astype(np.int32)

    def initial_distribution(self):
        # Integrate linear interpolation weights over Uniform(.1,.9) exactly.
        weights=initial_weights(self.bins)
        intercept=np.zeros(self.n,bool)
        for t in range(1,self.size):
            intercept|=(self.key[:,1]>t)&(np.abs(self.key-np.array([0,t])-self.defender).sum(1)<=2*t)
        mask=((self.key[:,1]>=starting_row(self.size))&
              (self.dec[:,1]>=starting_row(self.size))&
              (self.keypos!=self.decpos)&(self.keypos!=self.defpos)&
              (self.decpos!=self.defpos)&intercept)
        if self.size==6 and SMALL_CAPTURE_RADIUS>0:
            mask&=(np.abs(self.key-self.defender).sum(1)<=SMALL_CAPTURE_RADIUS)&(np.abs(self.dec-self.defender).sum(1)<=SMALL_CAPTURE_RADIUS)
        rho=mask*weights[self.ib];return rho/rho.sum()

    def build(self):
        size=self.size;bins=self.bins
        self.lower=np.empty((self.n,16),np.int32);self.blend=np.empty((self.n,16))
        # Motion likelihood depends only on physical position and realized
        # move, not belief, the other attacker, or defender position.
        pos=self.xy(np.arange(size*size))
        dest=np.clip(pos[:,None,:]+MOVES,0,size-1)
        destination=dest[:,:,0]+size*dest[:,:,1]
        likelihood=np.empty((size*size,4,2))
        for move in range(4):
            likelihood[:,move]=(dest==dest[:,move,None,:]).all(2)@MODE_PROBS.T
        for m in range(16):
            k=destination[self.keypos,m//4];c=destination[self.decpos,m%4]
            lk=likelihood[self.keypos,m//4];lc=likelihood[self.decpos,m%4]
            posterior=self.logodds+np.log(lk[:,1])+np.log(lc[:,0])-np.log(lk[:,0])-np.log(lc[:,1])
            posterior=np.clip(posterior,-LIMIT,LIMIT)
            lower=np.clip(np.searchsorted(self.log_grid,posterior,side="right")-1,0,bins-2)
            self.lower[:,m]=self.pack(k,c,self.defpos,lower)
            self.blend[:,m]=(posterior-self.log_grid[lower])/(self.log_grid[lower+1]-self.log_grid[lower])
        self.next=np.empty((self.n,9),np.int32)
        self.reward=np.empty((self.n,9));self.alive=np.empty((self.n,9))
        self.timeout=np.empty((self.n,9))
        dec_hit=self.decpos==self.defpos
        self.immediate_reward=-4.*self.key[:,1]+DIVERSION_WEIGHT*dec_hit*np.abs(self.key-self.defender).sum(1)
        for o,move in enumerate(DEFENDER_MOVES):
            first=np.clip(self.defender+np.sign(move),0,size-1)
            final=np.clip(self.defender+move,0,size-1)
            k1=(first==self.key).all(1);c1=(first==self.dec).all(1)
            stopped=k1|c1
            destination=np.where(stopped[:,None],first,final)
            kh=(destination==self.key).all(1);ch=(destination==self.dec).all(1)
            done=kh|ch
            terminal_reward=-4.*self.key[:,1]+DIVERSION_WEIGHT*ch*np.abs(self.key-destination).sum(1)
            self.reward[:,o]=np.where(done,terminal_reward,0)
            self.timeout[:,o]=terminal_reward
            self.alive[:,o]=~done
            self.next[:,o]=self.pack(self.keypos,self.decpos,destination[:,0]+size*destination[:,1],self.ib)

    @staticmethod
    def ptr(a):return a.ctypes.data_as(ctypes.c_void_p)

    def step(self,v,pi,kappa,t):
        if os.environ.get('SIM_BELLMAN_DEVICE'):
            from gpu_bellman import step
            return step(self,v,pi,kappa,t)
        fields=v.shape[1]
        p=np.ascontiguousarray((1-kappa)*self.p0[t]+kappa*self.p1[t])
        dp=np.ascontiguousarray(self.p1[t]-self.p0[t])
        reward=self.timeout if t==self.horizon-1 else self.reward
        alive=np.zeros_like(self.alive) if t==self.horizon-1 else self.alive
        mid=np.empty_like(v);out=np.empty_like(v)
        self.lib.defender_step(self.n,fields,*map(self.ptr,[self.next,reward,alive,p,dp,
             self.terminal,self.immediate_reward,v]),self.gamma,self.ptr(mid))
        self.lib.attacker_step(self.n,fields,*map(self.ptr,[self.lower,self.blend,self.laws,
             pi,self.feature,self.terminal,mid,out]))
        return out

    def evaluate(self,theta,kappa=.7,fields=17,depth=None,boundary=None,penalty=0.,rho=None,save_values=False):
        pi=np.ascontiguousarray(probabilities(theta,self.feature))
        depth=self.horizon if depth is None else depth
        v=np.zeros((self.n,fields))
        if boundary is not None:v[:,0]=boundary
        elif depth<self.horizon:
            b=np.zeros((self.n,1))
            for t in reversed(range(depth,self.horizon)):b=self.step(b,pi,kappa,t)
            v[:,0]=b[:,0]
        values={depth:v[:,0].copy()} if save_values else None
        for t in reversed(range(depth)):
            v=self.step(v,pi,kappa,t)
            if save_values:values[t]=v[:,0].copy()
        r=(self.initial if rho is None else rho)@v
        r[0]-=.5*penalty*np.dot(theta,theta)
        if fields>1:r[1:4]-=penalty*np.asarray(theta)
        if fields==17:r[5:14]-=(penalty*np.eye(3)).ravel()
        return r,v,values
