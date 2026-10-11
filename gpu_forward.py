"""Opt-in double-precision forward on a declared, separately audited device."""
import os,threading
import torch
_LOCK=threading.Lock();_CACHE={};_READY=False

def forward(weights,x):
    global _READY
    device=torch.device(os.environ['SIM_FORWARD_DEVICE'])
    key=tuple(id(w) for w in weights)
    with _LOCK:
        if not _READY:
            torch.cuda.set_per_process_memory_fraction(.9 if os.environ.get('SIM_MODEL_GPU_RESIDENT') else .08,device);_READY=True
        if key not in _CACHE:
            if len(_CACHE)>=8:_CACHE.pop(next(iter(_CACHE)))
            _CACHE[key]=(tuple(weights),[torch.as_tensor(w,dtype=torch.float64,device=device) for w in weights])
        ws=_CACHE[key][1]
    with torch.inference_mode():
        z=torch.as_tensor(x,dtype=torch.float64,device=device)
        for i in range(0,len(ws),2):
            z=z@ws[i].T+ws[i+1]
            if i<len(ws)-2:z=z.tanh()
        return z.cpu().numpy()
