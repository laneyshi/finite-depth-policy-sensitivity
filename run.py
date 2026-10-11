"""Fresh small-instance figures with an explicit belief-grid convergence audit."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
os.environ.update(DESIGNED_DIVERSION_WEIGHT='.25',DESIGNED_SMALL_START_FRACTION='.5',
                  DESIGNED_SMALL_CAPTURE_RADIUS='2',DESIGNED_REWARD_SCALE='1',DESIGNED_ENCOUNTER_POOL='')
from pathlib import Path
import json,hashlib,subprocess,sys,argparse,shutil
import numpy as np
import torch
from scipy.optimize import minimize
from gpu_exact_model import Model
from train import train, pretrain
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def reference(bins,start):
    dest=OUT/f'b{bins}';dest.mkdir(exist_ok=True)
    if start:
        theta=np.load(start)['theta'][:2]
    else:theta=torch.load(OUT/'ppo_warm.pt',weights_only=True)['actor']['theta'].numpy().astype(float)
    model=Model(bins=bins)
    # Fine meshes start at an already polished neighboring-grid reference.
    # Newton's derivative residual avoids objective line-search decisions below
    # floating-point resolution; only negative-definite local Hessians qualify.
    r,_,_=model.evaluate(np.r_[theta,0.])
    newton_history=[]
    if np.linalg.norm(r[1:3])<1e-4:
        for iteration in range(4):
            h=r[5:14].reshape(3,3)[:2,:2]
            norm=float(np.linalg.norm(r[1:3]))
            newton_history.append(dict(iteration=iteration,theta=theta.tolist(),gradient_norm=norm))
            print('LOCAL_NEWTON',bins,newton_history[-1],flush=True)
            if norm<1e-8:break
            if max(np.linalg.eigvalsh(h))>=0:break
            candidate=theta-np.linalg.solve(h,r[1:3])
            if np.linalg.norm(candidate-theta)>.01:break
            trial,_,_=model.evaluate(np.r_[candidate,0.])
            if np.linalg.norm(trial[1:3])>=norm:break
            theta,r=candidate,trial
    class GradientConverged(Exception):
        def __init__(self,x):self.x=x.copy()
    def fg(x):
        r,_,_=model.evaluate(np.r_[x,0.],fields=4)
        print('REFINE',bins,float(r[0]),float(np.linalg.norm(r[1:3])),x,flush=True)
        # Stop on the independently computed derivative, before a line search
        # rejects an adequate point because of ~1e-13 objective roundoff.
        if np.linalg.norm(r[1:3])<1e-8:raise GradientConverged(x)
        return -r[0],-r[1:3]
    if np.linalg.norm(r[1:3])<1e-8:
        success=True;theta=np.r_[theta,0.]
    else:
        try:
            sol=minimize(fg,theta,jac=True,method='BFGS',options=dict(gtol=1e-9,maxiter=80))
            theta=sol.x;success=bool(sol.success)
        except GradientConverged as sol:
            theta=sol.x;success=True
        theta=np.r_[theta,0.];r,_,_=model.evaluate(theta)
    h=r[5:14].reshape(3,3)[:2,:2];m=r[14:16];s=-np.linalg.solve(h,m)
    np.savez_compressed(dest/'reference.npz',theta=theta,H=h,M=m,S=s,jet=r)
    report=dict(bins=bins,gamma=.9,horizon=15,kappa=.7,theta=theta.tolist(),value=float(r[0]),
        gradient_norm=float(np.linalg.norm(r[1:3])),eigenvalues=np.linalg.eigvalsh(h).tolist(),
        diversion_weight=.25,small_start_fraction=.5,small_capture_radius=2,
        policy='Two-coefficient identifiable softmax; trained with PPO then explicit deterministic reference polishing',
        reference_polishing='Local Newton derivative refinement when near-stationary, otherwise BFGS on finite-grid Bellman objective; no entropy/ridge added',
        local_newton_history=newton_history,
        PPO_only_claim=False,continuous_belief_exact_claim=False,optimizer_success=success,
        gradient_early_stop=1e-8,belief_representation='Nonuniform log-odds lattice [-86.4,86.4], fine central [-21.6,21.6] and coarse tails; stable Bayes updates')
    (dest/'report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
    if report['gradient_norm']>1e-6 or max(report['eigenvalues'])>=0:
        raise RuntimeError('Reference stationarity/negative curvature check failed; retain results for diagnosis')
    shutil.copy2(OUT/'ppo_warm.pt',dest/'ppo_warm.pt')

def grid_audit(bins,ref):
    z=np.load(ref);model=Model(bins=bins);r,_,_=model.evaluate(z['theta'])
    h=r[5:14].reshape(3,3)[:2,:2];m=r[14:16];s=-np.linalg.solve(h,m)
    errors={name:float(np.linalg.norm(value-z[name],2)/np.linalg.norm(value,2)) for name,value in [('H',h),('M',m),('S',s)]}
    np.savez_compressed(Path(ref).parent/f'grid_audit_{bins}.npz',theta=z['theta'],H=h,M=m,S=s,jet=r)
    result=dict(reference=str(ref),refined_bins=bins,common_theta=z['theta'].tolist(),relative_changes=errors,
                threshold=2.5e-4,passed=max(errors.values())<=2.5e-4,
        gradient_norm=float(np.linalg.norm(r[1:3])),eigenvalues=np.linalg.eigvalsh(h).tolist(),
        caveat='Observed log-odds mesh-refinement check, not a rigorous bound or mathematically exact continuous-belief expectation')
    (Path(ref).parent/f'grid_audit_{bins}.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

def run_stage(args,log):
    with (HERE/log).open('w') as stream:
        subprocess.run([sys.executable,'-u',str(HERE/'run.py'),*args],stdout=stream,stderr=subprocess.STDOUT,check=True)

def all_stages():
    OUT.mkdir(exist_ok=True)
    if (OUT/'b1217/reference.npz').exists():
        raise RuntimeError('Existing experiment. Resume individual stages or use a fresh results directory.')
    run_stage(['reference','--bins','1217'],'reference.log')
    run_stage(['audit','--bins','1825','--start',str(OUT/'b1217/reference.npz')],'audit.log')
    audit=json.loads((OUT/'b1217/grid_audit_1825.json').read_text())
    if not audit['passed']:raise RuntimeError('Mesh refinement audit failed')
    for stage in ['prepare','baseline','plot']:
        run_stage([stage,'--bins','1217'],stage+'.log')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','endpoints','ppo','reference','audit','prepare','baseline','plot'])
    p.add_argument('--bins',type=int,default=1217);p.add_argument('--start');a=p.parse_args()
    if a.stage=='all':
        try:all_stages()
        except Exception:(HERE/'STATUS.txt').write_text('FAILED: inspect stage logs\n');raise
    elif a.stage=='endpoints':
        OUT.mkdir(exist_ok=True)
        pretrain(6)
        # Small-kernel tests reuse the size-6 weights on normalized coordinates.
        shutil.copy2(OUT/'endpoints_g6.pt', OUT/'endpoints_g3.pt')
    elif a.stage=='ppo':
        import belief, uniform_belief
        belief.project=uniform_belief.project
        train(OUT/'ppo_warm',.7,steps=1000000,seed=891000,size=6,horizon=15,gamma=.9,
              actor_kind='structured',actor_lr=.003,bins=1025)
    elif a.stage=='reference':reference(a.bins,a.start)
    elif a.stage=='audit':grid_audit(a.bins,a.start)
    elif a.stage=='prepare':
        from small_validation import prepare
        import fcntl
        dest=OUT/f'b{a.bins}'
        with (dest/'prepare.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (dest/'conditional.npz').exists():prepare(dest,a.bins)
    elif a.stage=='baseline':
        from small_validation import baseline
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=int(os.environ.get("SIM_WORKERS","10")),mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(baseline,[(str(OUT/f'b{a.bins}'),seed) for seed in range(10)]):print(result,flush=True)
    elif a.stage=='plot':
        from report_small import run
        run(OUT/f'b{a.bins}')
