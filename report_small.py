"""Plot real arrays, retaining both plain and variance-reduced LR baselines."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(folder):
    out=Path(folder);figs=out/'figures';figs.mkdir(exist_ok=True)
    r=np.load(out/'reference.npz');h,m,s=r['H'],r['M'],r['S']
    b=[np.load(f) for f in sorted(out.glob('baseline*.npz'))]
    assert len(b)==10
    def errors(jet):
        hh=jet[...,5:14].reshape(*jet.shape[:-1],3,3)[...,:2,:2]
        mm=jet[...,14:16];ss=-np.linalg.solve(hh,mm[...,None])[...,0]
        return np.stack([np.linalg.norm(hh-h,ord=2,axis=(-2,-1))/np.linalg.norm(h,2),
                         np.linalg.norm(mm-m,axis=-1)/np.linalg.norm(m),
                         np.linalg.norm(ss-s,axis=-1)/np.linalg.norm(s)],axis=-1)
    if (out/'ideal_expectation.npz').exists():
        exact=np.load(out/'ideal_expectation.npz')['jet']
    else:
        z=np.load(out/'conditional.npz')
        exact=np.einsum('tnj,n->tj',z['ideal'],z['weights'])
    ideal=errors(exact);sampled=errors(np.stack([x['raw'][-1] for x in b]))
    assert np.isfinite(ideal).all() and np.isfinite(sampled).all()
    assert np.max(ideal[-1])<1e-12, 'Full-depth truncation check failed'
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for j,label in enumerate(['H','M','S']):
        axes[0].semilogy(range(1,15),ideal[:14,j],'-o',ms=3,label=label)
        med=np.median(sampled[:,:,j],axis=0);lo,hi=np.quantile(sampled[:,:,j],[.25,.75],axis=0)
        axes[1].semilogy(range(1,16),med,'-o',ms=3,label=label)
        axes[1].fill_between(range(1,16),lo,hi,alpha=.15)
    for ax,title in zip(axes,['Exact sums in audited belief-grid model','PPO-derived critic; N = 1,000,000 (10 seeds)']):
        ax.set(title=title,xlabel='Propagation depth',ylabel='Relative error');ax.grid(alpha=.2);ax.legend()
    axes[0].set_xlim(.5,15.5)
    axes[0].text(.03,.04,'T=15: truncation vanishes\n(roundoff below 1e-12)',transform=axes[0].transAxes,fontsize=8,
                 bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    fig.tight_layout();fig.savefig(figs/'error_depth.pdf');fig.savefig(figs/'error_depth.png',dpi=180);plt.close(fig)
    raw=errors(np.stack([x['raw'] for x in b]));n=b[0]['budgets']
    lh=np.stack([x['h'] for x in b]);lm=np.stack([x['m'] for x in b])
    ch=np.stack([x['h_cv'] for x in b]);cm=np.stack([x['m_cv'] for x in b])
    lr=np.stack([np.linalg.norm(lh-h,ord=2,axis=(-2,-1))/np.linalg.norm(h,2),
                 np.linalg.norm(lm-m,axis=-1)/np.linalg.norm(m)],axis=-1)
    cv=np.stack([np.linalg.norm(ch-h,ord=2,axis=(-2,-1))/np.linalg.norm(h,2),
                 np.linalg.norm(cm-m,axis=-1)/np.linalg.norm(m)],axis=-1)
    conclusions={}
    for supplemental in [False,True]:
        fig,axes=plt.subplots(2,3,figsize=(12,6.7))
        for col,t in enumerate([3,5,7]):
            for row,label in enumerate(['H','M']):
                ax=axes[row,col]
                series=[(raw[:,:,t-1,row],'Proposed method','-o'),
                        (lr[:,:,row],'Trajectory LR','--s')]
                if supplemental:
                    series.append((cv[:,:,row],'LR + critic control (additional baseline)','-.'))
                for a,name,style in series:
                    med=np.median(a,axis=0);lo,hi=np.quantile(a,[.25,.75],axis=0)
                    ax.loglog(n,med,style,ms=3,label=name);ax.fill_between(n,lo,hi,alpha=.1)
                ax.set(title=f'{label}, T={t}',xlabel='Sample size N',ylabel='Relative error');ax.grid(alpha=.2)
                if row==0 and col==0:ax.legend(fontsize=7)
                conclusions[f'{label}_T{t}_beats_plain_LR_all_N']=bool(np.all(np.median(raw[:,:,t-1,row],0)<np.median(lr[:,:,row],0)))
                conclusions[f'{label}_T{t}_beats_controlled_LR_all_N']=bool(np.all(np.median(raw[:,:,t-1,row],0)<np.median(cv[:,:,row],0)))
        fig.suptitle('Relative estimation error versus sample size'+(' — additional LR baseline' if supplemental else ''),fontsize=13)
        fig.text(.5,.015,'N: initial states for Proposed method; complete trajectories for LR. Median and interquartile range over 10 seeds.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.045,1,.955])
        name='comparison_supplementary' if supplemental else 'comparison'
        fig.savefig(figs/(name+'.pdf'));fig.savefig(figs/(name+'.png'),dpi=180);plt.close(fig)
    from bounds import Constants, derivative_bounds, sensitivity_bound
    # Conservative constants, derived before inspecting any error curves:
    # |reward| <= 4*5 + .25*(2*5); |features_i|<=1;
    # ||score||<=sqrt(2), ||log-policy Hessian||<=.5, and
    # ||d P / d kappa||_TV <= ||psi1-psi0||_1 <= 2.
    constants=Constants(22.5,np.sqrt(2),2.5,2.)
    eh,em=derivative_bounds(15,.9,constants)
    actual_h=ideal[:,0]*np.linalg.norm(h,2);actual_m=ideal[:,1]*np.linalg.norm(m)
    curvature=-np.linalg.eigvalsh(h)[-1]
    bound_s=sensitivity_bound(eh,em,curvature,np.linalg.norm(s))
    posterior_s=sensitivity_bound(actual_h,actual_m,curvature,np.linalg.norm(s))
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for ax,actual,bound,label in zip(axes[:2],[actual_h,actual_m],[eh,em],['H','M']):
        ax.semilogy(range(1,15),actual[:14],'-o',ms=3,label='Actual truncation error')
        ax.semilogy(range(1,15),bound[:14],'--',label='Uniform theorem bound')
        ax.set(title=label,xlabel='Depth',ylabel='Absolute norm');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[2].semilogy(range(1,15),ideal[:14,2]*np.linalg.norm(s),'-o',ms=3,label='Actual sensitivity error')
    valid=np.isfinite(posterior_s)&(np.arange(15)<14)
    axes[2].semilogy(np.arange(1,16)[valid],posterior_s[valid],'--s',ms=3,label='A posteriori perturbation bound')
    axes[2].set(title='S (conditional bound)',xlabel='Depth',ylabel='Absolute norm');axes[2].grid(alpha=.2);axes[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(figs/'theorem_bounds.pdf');fig.savefig(figs/'theorem_bounds.png',dpi=180);plt.close(fig)
    report=dict(reference=json.loads((out/'report.json').read_text()),
        theorem_constants=vars(constants),full_depth_truncation_errors=ideal[-1].tolist(),
        full_depth_sampling_errors=np.median(sampled[:,-1],axis=0).tolist(),
        ideal_errors=ideal.tolist(),conclusions=conclusions,
        derivative_bounds_pass=bool(np.all(actual_h<=eh+1e-11) and np.all(actual_m<=em+1e-11)),
        uniform_sensitivity_condition_depths=(np.flatnonzero(np.isfinite(bound_s))+1).tolist(),
        posterior_sensitivity_condition_depths=(np.flatnonzero(np.isfinite(posterior_s))+1).tolist(),
        note='Plain LR and critic-controlled LR both retained. No claim of equal model computation. Uniform and a posteriori sensitivity bounds distinguished.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(conclusions,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();run(a.folder)
