"""Chunked float64 CUDA Bellman kernels; CPU implementation remains the oracle."""
import os,ctypes,threading
from pathlib import Path
import numpy as np
import torch
_LOCK=threading.Lock();_LIB=None

def step(model,v,pi,kappa,t):
    global _LIB
    with _LOCK:
        device=torch.device(os.environ['SIM_BELLMAN_DEVICE'])
        torch.cuda.set_per_process_memory_fraction(.65,device)
        if _LIB is None:
            _LIB=ctypes.CDLL(str(Path(__file__).with_name('bellman_gpu.so')))
            ptr=ctypes.c_void_p
            _LIB.defender_step.argtypes=[ctypes.c_int,ctypes.c_int]+[ptr]*8+[ctypes.c_double,ptr,ctypes.c_size_t]
            _LIB.attacker_step.argtypes=[ctypes.c_int,ctypes.c_int]+[ptr]*8+[ctypes.c_size_t]
            _LIB.defender_step.restype=ctypes.c_int;_LIB.attacker_step.restype=ctypes.c_int
            _LIB.select_device.argtypes=[ctypes.c_int];_LIB.select_device.restype=ctypes.c_int
        error=_LIB.select_device(device.index)
        if error:raise RuntimeError(f'CUDA device error {error}')
        stream=torch.cuda.current_stream(device).cuda_stream
        def tensor(a):return torch.as_tensor(np.ascontiguousarray(a),device=device)
        def pointers(arrays):return [ctypes.c_void_p(a.data_ptr()) for a in arrays]
        fields=v.shape[1];n=model.n;chunk=262144
        value=tensor(v);mid=torch.empty_like(value)
        reward=model.timeout if t==model.horizon-1 else model.reward
        for start in range(0,n,chunk):
            end=min(start+chunk,n);sl=slice(start,end)
            params=[tensor(model.next[sl]),tensor(reward[sl]),
                    torch.zeros((end-start,9),dtype=torch.float64,device=device) if t==model.horizon-1 else tensor(model.alive[sl]),
                    tensor((1-kappa)*model.p0[t][sl]+kappa*model.p1[t][sl]),
                    tensor(model.p1[t][sl]-model.p0[t][sl]),tensor(model.terminal[sl]),tensor(model.immediate_reward[sl]),value]
            error=_LIB.defender_step(end-start,fields,*pointers(params),model.gamma,ctypes.c_void_p(mid[sl].data_ptr()),stream)
            if error:raise RuntimeError(f'CUDA defender kernel error {error}')
        del params,value
        out=np.empty_like(v);laws=tensor(model.laws)
        for start in range(0,n,chunk):
            end=min(start+chunk,n);sl=slice(start,end);dest=torch.empty((end-start,fields),dtype=torch.float64,device=device)
            params=[tensor(model.lower[sl]),tensor(model.blend[sl]),laws,tensor(pi[sl]),tensor(model.feature[sl]),tensor(model.terminal[sl]),mid,dest]
            error=_LIB.attacker_step(end-start,fields,*pointers(params),stream)
            if error:raise RuntimeError(f'CUDA attacker kernel error {error}')
            out[sl]=dest.cpu().numpy()
        return out
