"""Full-network PPO and independent defender endpoint pretraining."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from environment import Environment, sample, teacher, DIVERSION_WEIGHT, ENCOUNTER_POOL
from networks import Network, StructuredActor, CircleActor, SharedActor, CIRCLE_RADII, numpy_net, forward, softmax, endpoints, load_endpoints, ROOT

def pretrain(size, updates=2500):
    torch.manual_seed(12000+size); rng=np.random.default_rng(12000+size)
    nets=[Network(6,9),Network(7,9)]; reports=[]
    for aware, net in enumerate(nets):
        opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        for i in range(updates):
            e=Environment(512,rng,size,15 if size==6 else 60)
            # Cover all physical states, not just the starting distribution.
            e.a=rng.integers(size,size=(512,2,2));e.d=rng.integers(size,size=(512,2))
            e.b=rng.uniform(0,1,512)
            x=torch.tensor(e.observations('aware' if aware else 'blind'),dtype=torch.float32)
            target=torch.tensor(teacher(e,aware),dtype=torch.float32)
            loss=-(target*net(x).log_softmax(1)).sum(1).mean()
            opt.zero_grad();loss.backward();opt.step()
        e=Environment(10000,rng,size,15 if size==6 else 60)
        e.a=rng.integers(size,size=(10000,2,2));e.d=rng.integers(size,size=(10000,2));e.b=rng.uniform(0,1,10000)
        p=softmax(forward(numpy_net(net),e.observations('aware' if aware else 'blind')))
        target=teacher(e,aware)
        reports.append(dict(kind='aware' if aware else 'blind',
            heldout_KL=float(np.mean(np.sum(target*np.log(np.maximum(target,1e-300)/p),axis=1)))))
        print('endpoint',size,reports[-1],flush=True)
    out=ROOT/f'results/endpoints_g{size}.pt';out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(blind=nets[0].state_dict(),aware=nets[1].state_dict(),size=size,updates=updates),out)
    out.with_suffix('.json').write_text(json.dumps(reports,indent=2))

def collect(actor,critic,blind,aware,kappa,n,rng,size=20,horizon=60,gamma=.99,training=True,bins=None):
    e=Environment(n,rng,size,horizon)
    def project():
        if bins is not None:
            from belief import project as project_logodds
            project_logodds(e,bins)
    project()
    xs=[];acts=[];probs=[];rewards=[];masks=[];values=[]
    for t in range(horizon):
        x=e.observations();p=softmax(forward(actor,x));a=sample(p,rng);live=~e.done.copy()
        if training:
            xs.append(x);acts.append(a);probs.append(p[np.arange(n),a]);masks.append(live)
            values.append(forward(critic,x).ravel()*4*(size-1))
        e.attacker_phase(a)
        project()
        p0,p1=endpoints(e,blind,aware)
        e.defender_phase(sample((1-kappa)*p0+kappa*p1,rng))
        rewards.append(e.reward.copy());e.reward.fill(0)
        if e.done.all():break
    rewards=np.asarray(rewards);returns=np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        returns[t]=rewards[t]+(gamma*returns[t+1] if t+1<len(rewards) else 0)
    if not training:return returns[0]
    masks=np.asarray(masks);values=np.asarray(values)
    weights=np.broadcast_to(gamma**np.arange(len(rewards))[:,None],masks.shape)
    return dict(obs=np.asarray(xs)[masks],action=np.asarray(acts)[masks],oldp=np.asarray(probs)[masks],
        target=returns[masks]/(4*(size-1)),advantage=(returns-values)[masks],weight=weights[masks],
        steps=int(masks.sum()),mean_return=float(returns[0].mean()))

def train(out,kappa,steps=2000000,seed=0,reference=None,initial_actor=None,evaluate=False,
          size=20,horizon=60,gamma=.99,actor_kind='neural',actor_lr=3e-4,bins=None,
          initial_eval_episodes=2000,random_initial_actor=False):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    blind,aware=load_endpoints(size)
    actor={'structured':StructuredActor,'circle':CircleActor,'shared':SharedActor}.get(actor_kind,lambda:Network(9,4))()
    critic=Network(9,1)
    if random_initial_actor:
        assert actor_kind=='shared' and reference is None and initial_actor is None
        with torch.no_grad():actor.theta.fill_(float(np.random.default_rng(seed+740000).normal()))
    if reference:
        ck=torch.load(reference,map_location='cpu',weights_only=True)
        actor.load_state_dict(ck['actor']);critic.load_state_dict(ck['critic'])
    if initial_actor:actor.load_state_dict(initial_actor)
    initial_parameters=[p.detach().numpy().tolist() for p in actor.parameters()]
    opt=torch.optim.Adam(actor.parameters(),lr=actor_lr)
    vopt=torch.optim.Adam(critic.parameters(),lr=1e-3)
    total=0;logs=[];curves=[];start=time.time()
    def ev():
        count=initial_eval_episodes if not curves else 2000
        erng=np.random.default_rng(910000+seed*101+len(curves))
        ret=np.concatenate([collect(numpy_net(actor),None,blind,aware,kappa,min(2000,count-i),
                    erng,size,horizon,gamma,False,bins) for i in range(0,count,2000)])
        if not curves:np.save(str(out)+'.initial_returns.npy',ret)
        curves.append([total,float(ret.mean()),float(ret.std(ddof=1)/np.sqrt(len(ret)))])
        print(out.name,'eval',curves[-1],flush=True)
    if evaluate:ev()
    milestones=list(range(20000,steps+1,20000)) if evaluate else [steps]
    for target in milestones:
        while total<target:
            reference_anneal=os.environ.get('DESIGNED_REFERENCE_ANNEAL','0')=='1' and not evaluate
            n=min(512 if reference_anneal else 128,max(1,(target-total)//horizon))
            data=collect(numpy_net(actor),numpy_net(critic),blind,aware,kappa,n,rng,size,horizon,gamma,bins=bins)
            total+=data['steps']
            if reference_anneal:
                for group in opt.param_groups:group['lr']=actor_lr*.5*(1+np.cos(np.pi*min(total/steps,1)))
            z={k:torch.tensor(v,dtype=torch.long if k=='action' else torch.float32)
                for k,v in data.items() if k not in ('steps','mean_return')}
            adv=z['advantage']/(z['advantage'].std(unbiased=False)+1e-6)
            # Exploration only in the initial half of reference/scratch training;
            # no entropy in derivative objective. All transfer arms identical.
            entropy=.005*max(0.,1-2*total/steps) if reference is None else 0.
            for epoch in range(4):
                for ix in torch.randperm(len(adv)).split(512):
                    lp=actor(z['obs'][ix]).log_softmax(1)
                    ratio=lp.gather(1,z['action'][ix,None]).ravel().exp()/z['oldp'][ix]
                    objective=torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix])
                    ent=-(lp.exp()*lp).sum(1)
                    loss=-(z['weight'][ix]*(objective+entropy*ent)).mean()
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);opt.step()
                    vloss=(critic(z['obs'][ix]).ravel()-z['target'][ix]).square().mean()
                    vopt.zero_grad();vloss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);vopt.step()
            logs.append([total,data['mean_return']])
            if len(logs)%100==0:print(out.name,'train',total,logs[-1][1],flush=True)
        if evaluate:ev()
    torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),kappa=kappa,seed=seed,
                    size=size,horizon=horizon,gamma=gamma,steps=total,actor_kind=actor_kind,
                    diversion_weight=DIVERSION_WEIGHT,circle_radii=CIRCLE_RADII.tolist(),
                    encounter_pool=ENCOUNTER_POOL,belief_bins=bins),out.with_suffix('.pt'))
    np.savez_compressed(out.with_suffix('.npz'),training=logs,curve=curves)
    out.with_suffix('.json').write_text(json.dumps(dict(kappa=kappa,seed=seed,steps=total,
        seconds=time.time()-start,reference=str(reference),actor_parameter_count=sum(p.numel() for p in actor.parameters()),
        initial_parameters=initial_parameters,initial_eval_episodes=initial_eval_episodes,
        random_initial_actor=random_initial_actor),indent=2))

def reference(k):
    out=ROOT/f'results/references/k{round(k*10):02d}'
    if not out.with_suffix('.pt').exists():train(out,k,seed=13000+round(k*100))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--endpoints',action='store_true')
    p.add_argument('--references',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.endpoints:
        with ProcessPoolExecutor(max_workers=2) as ex:list(ex.map(pretrain,[6,20]))
    if a.references:
        with ProcessPoolExecutor(max_workers=4) as ex:list(ex.map(reference,[.5,.6,.7,1.]))
    if a.smoke:train(ROOT/'results/smoke',.7,steps=2000,seed=11)
