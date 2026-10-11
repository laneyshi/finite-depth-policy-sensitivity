import run,os,time,json
import numpy as np
from exact_model import Model as Original
from gpu_exact_model import Model,MAP
os.environ['SIM_COMPACT_BACKEND']='cpu'
a=Original(bins=39,size=3,horizon=3);b=Model(bins=39,size=3,horizon=3);theta=np.array([-.13,-2.,0.]);rows=[]
for fields in [1,4,17]:
 r,_,_=a.evaluate(theta,fields=fields);t=time.time();s,_,_=b.evaluate(theta,fields=fields)
 keep=[0] if fields==1 else [0,1,2] if fields==4 else MAP+[8]
 assert np.allclose(r[keep],s[keep],rtol=1e-10,atol=1e-11)
 rows.append(dict(fields=fields,max_difference=float(np.max(abs(r[keep]-s[keep]))),compact_seconds=time.time()-t))
(run.HERE/'COMPACT_CPU_VALIDATION.json').write_text(json.dumps(rows,indent=2));print(rows)
