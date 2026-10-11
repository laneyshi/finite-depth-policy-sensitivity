"""Exact truth, fitted critic, initial-state sampling, plain and controlled LR."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import torch
from exact_model import probabilities, features
from gpu_exact_model import Model
from networks import ROOT, Network, load_endpoints, endpoints, numpy_net, forward
from environment import Environment, sample, DIVERSION_WEIGHT, SMALL_START_FRACTION, SMALL_CAPTURE_RADIUS
from belief import project, LIMIT, grid

BUDGETS=[100,300,1000,3000,10000,30000,100000,300000,1000000]

def initial(n,rng,bins):
    e=Environment(n,rng,6,15);sw=e.role==1;e.a[sw]=e.a[sw,::-1];e.b[sw]=1-e.b[sw];e.role.fill(0)
    e.logodds[sw]*=-1
    project(e,bins);return e

def index(e,bins):
    p=e.a[:,:,0]+6*e.a[:,:,1];d=e.d[:,0]+6*e.d[:,1]
    return (((p[:,0]*36+p[:,1])*36+d)*bins+np.searchsorted(grid(bins),e.logodds))

def jets(theta,e,action):
    f=features(e.a[:,0],e.a[:,1],e.d,e.b,6);p=probabilities(theta,f)
    f=f[:,:2];p=p[:,:2]
    score=(np.eye(4)[action,:2]-p)*f
    h=-(np.eye(2)[None]*p[:,:,None]-p[:,:,None]*p[:,None,:])*f[:,:,None]*f[:,None,:]
    return score,h

def episodes(theta,blind,aware,bins,n,rng,critic=None):
    e=initial(n,rng,bins);starts=index(e,bins)
    xs=[];mask=[];rewards=[]
    score=np.zeros((n,2));lh=np.zeros((n,2,2));sk=np.zeros(n)
    ch=np.zeros((n,2,2));cm=np.zeros((n,2));total=np.zeros(n)
    for t in range(15):
        live=~e.done.copy();x=e.observations();xs.append(x);mask.append(live)
        value=forward(critic,x).ravel()*20 if critic is not None else np.zeros(n)
        control=total+.9**t*value*live
        prev=score.copy();pk=sk.copy()
        f=features(e.a[:,0],e.a[:,1],e.d,e.b,6);p=probabilities(theta,f);a=sample(p,rng)
        g,h=jets(theta,e,a);score+=g*live[:,None];lh+=h*live[:,None,None]
        e.attacker_phase(a);project(e,bins)
        p0,p1=endpoints(e,blind,aware);pd=.3*p0+.7*p1;da=sample(pd,rng)
        sk+=(~e.done)*(p1-p0)[np.arange(n),da]/pd[np.arange(n),da]
        inc=score[:,:,None]*score[:,None,:]-prev[:,:,None]*prev[:,None,:]+h*live[:,None,None]
        ch+=control[:,None,None]*inc
        cm+=control[:,None]*(score*sk[:,None]-prev*pk[:,None])
        e.defender_phase(da);rewards.append(e.reward.copy());total+=.9**t*e.reward;e.reward.fill(0)
        if e.done.all():break
    rewards=np.array(rewards);returns=np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):returns[t]=rewards[t]+(.9*returns[t+1] if t+1<len(rewards) else 0)
    mask=np.array(mask)
    h=total[:,None,None]*(score[:,:,None]*score[:,None,:]+lh)
    m=total[:,None]*score*sk[:,None]
    return dict(h=h,m=m,h_cv=h-ch,m_cv=m-cm,starts=starts,
        obs=np.array(xs)[mask],target=returns[mask]/20)

def prepare(out,bins):
    out=Path(out);ref=np.load(out/'reference.npz');theta=ref['theta']
    meta=json.loads((out/'report.json').read_text())
    assert meta.get('diversion_weight',.25)==DIVERSION_WEIGHT
    assert meta.get('small_start_fraction',.7)==SMALL_START_FRACTION
    assert meta.get('small_capture_radius',0)==SMALL_CAPTURE_RADIUS
    model=Model(bins=bins);truth=model.evaluate(theta,fields=1,save_values=True)[2]
    # Exact value tables retained in memory; do not duplicate multi-GB arrays on disk.
    blind,aware=load_endpoints(6);torch.manual_seed(1819000);critic=Network(9,1)
    ck=out/'critic.pt'
    if ck.exists():critic.load_state_dict(torch.load(ck,weights_only=True)['critic'])
    else:
        critic.load_state_dict(torch.load(out/'ppo_warm.pt',weights_only=True)['critic'])
        rng=np.random.default_rng(1819000);obs=[];targets=[]
        for start in range(0,100000,4096):
            d=episodes(theta,blind,aware,bins,min(4096,100000-start),rng)
            obs.append(d['obs'].astype(np.float32));targets.append(d['target'].astype(np.float32))
        x=torch.tensor(np.concatenate(obs));y=torch.tensor(np.concatenate(targets))
        opt=torch.optim.Adam(critic.parameters(),lr=1e-3)
        for epoch in range(30):
            for ix in torch.randperm(len(x)).split(4096):
                loss=(critic(x[ix]).ravel()-y[ix]).square().mean()
                opt.zero_grad();loss.backward();opt.step()
            if epoch%5==4:print('critic epoch',epoch+1,'loss',loss.item(),flush=True)
        torch.save(dict(critic=critic.state_dict(),episodes=100000),ck)
    c=numpy_net(critic);ix=model.initial_indices
    depth_dir=out/'depth_tables';depth_dir.mkdir(exist_ok=True)
    def compute_depth(depth):
        dest=depth_dir/f'depth{depth:02d}.npz'
        if dest.exists():return str(dest)
        boundary=np.zeros(model.n)
        if depth<15:
            boundary=model.critic_boundary(c,depth)
        _,v,_=model.evaluate(theta,depth=depth,boundary=boundary);raw=v[ix].copy();del v
        del boundary
        _,v,_=model.evaluate(theta,depth=depth,boundary=truth[depth]);ideal=v[ix].copy();del v
        temporary=dest.with_name(dest.stem+'.partial.npz')
        np.savez_compressed(temporary,raw=raw,ideal=ideal);temporary.replace(dest)
        print('derivative depth',depth,flush=True)
        return str(dest)
    # Read-only model tables are shared; each C call owns all output buffers.
    # Preserve individual completed depths to resume without recalculation.
    with ThreadPoolExecutor(max_workers=1) as pool:
        list(pool.map(compute_depth,range(1,16)))
    raw=[];ideal=[]
    for depth in range(1,16):
        with np.load(depth_dir/f'depth{depth:02d}.npz') as z:
            raw.append(z['raw']);ideal.append(z['ideal'])
    np.savez_compressed(out/'conditional.partial.npz',raw=raw,ideal=ideal,indices=ix,
                        weights=model.initial_weights,bins=bins)
    (out/'conditional.partial.npz').replace(out/'conditional.npz')

def baseline(job):
    out,seed=job;out=Path(out);dest=out/f'baseline{seed:02d}.npz'
    if dest.exists():return str(dest)
    z=np.load(out/'conditional.npz');bins=int(z['bins']);ix=z['indices'];raw=z['raw'];ideal=z['ideal']
    theta=np.load(out/'reference.npz')['theta'];blind,aware=load_endpoints(6)
    net=Network(9,1);net.load_state_dict(torch.load(out/'critic.pt',weights_only=True)['critic']);c=numpy_net(net)
    meta=json.loads((out/'report.json').read_text())
    assert meta.get('diversion_weight',.25)==DIVERSION_WEIGHT
    assert meta.get('small_start_fraction',.7)==SMALL_START_FRACTION
    assert meta.get('small_capture_radius',0)==SMALL_CAPTURE_RADIUS
    rng=np.random.default_rng(1851000+seed);used=0
    sums={k:np.zeros(shape) for k,shape in [('h',(2,2)),('m',(2,)),('h_cv',(2,2)),('m_cv',(2,)),('raw',(15,17)),('ideal',(15,17))]}
    rows={k:[] for k in sums}
    for n in BUDGETS:
        while used<n:
            count=min(4096,n-used);d=episodes(theta,blind,aware,bins,count,rng,c)
            for k in ['h','m','h_cv','m_cv']:sums[k]+=d[k].sum(0)
            j=np.searchsorted(ix,d['starts']);assert np.all(ix[j]==d['starts'])
            sums['raw']+=raw[:,j].sum(1);sums['ideal']+=ideal[:,j].sum(1);used+=count
        for k in rows:rows[k].append(sums[k].copy()/n)
        print('baseline',seed,n,flush=True)
    np.savez_compressed(dest,**rows,budgets=BUDGETS)
    return str(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/small_d1');p.add_argument('--bins',type=int,default=9)
    p.add_argument('--prepare',action='store_true');p.add_argument('--baseline',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.out,a.bins)
    if a.baseline:
        with ProcessPoolExecutor(max_workers=10) as ex:
            for result in ex.map(baseline,[(a.out,s) for s in range(10)]):print(result,flush=True)
