import run,os,json,time
import numpy as np
from exact_model import Model as CPU
from gpu_exact_model import Model as GPU,MAP
from environment import Environment
from scipy.special import expit
from belief import LIMIT,initial_weights
os.environ.pop('SIM_FORWARD_DEVICE',None);os.environ.pop('SIM_BELLMAN_DEVICE',None)
a=CPU(bins=39,size=3,horizon=3);b=GPU(bins=39,size=3,horizon=3);theta=np.array([-.13,-2.,0.]);rows=[]
ix=np.flatnonzero(a.initial);assert np.array_equal(ix,b.initial_indices);assert np.allclose(a.initial[ix],b.initial_weights,atol=1e-15)
for fields in [1,4,17]:
 t=time.time();r,_,_=a.evaluate(theta,fields=fields);tc=time.time()-t
 t=time.time();s,_,_=b.evaluate(theta,fields=fields);tg=time.time()-t
 keep=[0] if fields==1 else [0,1,2] if fields==4 else MAP+[8]
 assert np.allclose(r[keep],s[keep],rtol=1e-10,atol=1e-11),(r,s)
 rows.append(dict(fields=fields,max_difference=float(np.max(abs(r[keep]-s[keep]))),cpu_seconds=tc,gpu_seconds=tg))
for depth in [1,2]:
 boundary=np.sin(np.arange(a.n)/100)
 r,_,_=a.evaluate(theta,depth=depth,boundary=boundary);s,_,_=b.evaluate(theta,depth=depth,boundary=boundary)
 assert np.allclose(r[MAP],s[MAP],rtol=1e-10,atol=1e-11)
e=Environment(1,np.random.default_rng(7),20,60);e.logodds[:]=50;e.b[:]=expit(e.logodds)
for _ in range(10):
 e.a[:]=[[[10,10],[15,10]]];e.d[:]=[0,0];e.done[:]=False
 e.attacker_phase(np.zeros(1,int),np.array([[0,1]]))
assert np.allclose(e.logodds,-4.) and np.allclose(e.b,expit(-4.))
for bins in [39,609,1217,1825,2433]:assert abs(initial_weights(bins).sum()-1)<1e-13
result=dict(jets=rows,only_declared_two_parameters=True,logodds_recovers_from_rounded_probability_one=True,initial_quadrature_mass=True)
(run.HERE/'COMPACT_GPU_VALIDATION.json').write_text(json.dumps(result,indent=2));print(result)
