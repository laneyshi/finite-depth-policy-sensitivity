"""Designed pursuit instance. Fixed dynamics/reward/rho0 across all kappa.

Design choices are explicit; this is not the unavailable original implementation.
"""
import os
import numpy as np
from scipy.special import expit

DIVERSION_WEIGHT=float(os.environ.get('DESIGNED_DIVERSION_WEIGHT','.25'))
SMALL_START_FRACTION=float(os.environ.get('DESIGNED_SMALL_START_FRACTION','.7'))
SMALL_CAPTURE_RADIUS=float(os.environ.get('DESIGNED_SMALL_CAPTURE_RADIUS','0'))
ENCOUNTER_POOL=os.environ.get('DESIGNED_ENCOUNTER_POOL','')
_POOL=None

def starting_row(size):
    return max(1,int((SMALL_START_FRACTION if size==6 else .7)*size))

MOVES = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])
DEFENDER_MOVES = np.array([[0, 2], [0, -2], [-2, 0], [2, 0],
                           [-1, 1], [1, 1], [-1, -1], [1, -1], [0, 0]])

def boltzmann(beta):
    x = beta*np.array([-1., 1., 0., 0.]); p = np.exp(x-x.max())
    return p/p.sum()

MODE_PROBS = np.array([boltzmann(.3), boltzmann(3.)])
LAWS = np.array([np.outer(MODE_PROBS[a//2], MODE_PROBS[a%2]).ravel() for a in range(4)])

def sample(p, rng):
    return np.minimum((rng.random(len(p))[:, None] > p.cumsum(1)).sum(1), p.shape[1]-1)

class Environment:
    def __init__(self, n, rng, size=20, horizon=60):
        self.n, self.rng, self.size, self.horizon, self.t = n, rng, size, horizon, 0
        # Attacker labels are physical labels; phi stays hidden from defender.
        self.a = np.stack([rng.integers(size, size=(n, 2)),
                           rng.integers(starting_row(size), size, size=(n, 2))], -1)
        self.role = rng.integers(2, size=n)
        self.d = rng.integers(size, size=(n, 2))
        # Fixed, kappa-independent rejection distribution: distinct agents and
        # defender able to intercept a straight goal run by the true key holder.
        key = self.a[np.arange(n), self.role]
        valid = self._valid(key)
        while not valid.all():
            j = np.flatnonzero(~valid)
            self.a[j, :, 0] = rng.integers(size, size=(len(j), 2))
            self.a[j, :, 1] = rng.integers(starting_row(size), size, size=(len(j), 2))
            self.d[j] = rng.integers(size, size=(len(j), 2))
            key = self.a[np.arange(n), self.role]
            valid = self._valid(key)
        self.b = rng.uniform(.1, .9, size=n)
        self.logodds=np.log(self.b)-np.log1p(-self.b)
        if size==20 and ENCOUNTER_POOL:
            global _POOL
            if _POOL is None:_POOL=np.load(ENCOUNTER_POOL)
            indices=rng.integers(len(_POOL['d']),size=n)
            self.a=_POOL['a'][indices].copy();self.d=_POOL['d'][indices].copy()
            sw=self.role==1;self.a[sw]=self.a[sw,::-1]
        self.done = np.zeros(n, bool)
        self.reward = np.zeros(n)
        self.reason = np.zeros(n, int)

    def _valid(self, key):
        valid = np.zeros(self.n, bool)
        for t in range(1, self.size):
            dest = key-np.array([0, t])
            # Cardinal speed two plus diagonal speed one: L1 reach <= 2t.
            valid |= (key[:, 1] > t) & (np.abs(dest-self.d).sum(1) <= 2*t)
        if self.size==6 and SMALL_CAPTURE_RADIUS>0:
            valid&=(np.abs(self.a-self.d[:,None]).sum(2)<=SMALL_CAPTURE_RADIUS).all(1)
        return valid & ~(self.a[:, 0] == self.a[:, 1]).all(1) & ~((self.a == self.d[:, None]).all(2).any(1))

    def observations(self, kind='attacker'):
        # Absolute and relative geometry; no hidden role in either endpoint.
        scale = self.size-1
        pos = np.column_stack([self.d/scale, (self.a[:, 0]-self.d)/scale,
                               (self.a[:, 1]-self.d)/scale])
        if kind == 'blind': return pos
        if kind == 'aware': return np.column_stack([pos, self.b])
        key = self.a[np.arange(self.n), self.role]
        dec = self.a[np.arange(self.n), 1-self.role]
        bk = np.where(self.role == 0, self.b, 1-self.b)
        # Key-ordered coordinates are for the attacker only.
        return np.column_stack([self.d/scale, (key-self.d)/scale, (dec-self.d)/scale,
                                bk, key[:, 1]/scale, np.full(self.n, self.t/self.horizon)])

    def _finish(self, mask, reason, decoy=None):
        mask = mask & ~self.done
        key = self.a[np.arange(self.n), self.role]
        value = -4.*key[:, 1].astype(float)
        if decoy is not None:
            value += DIVERSION_WEIGHT*decoy*np.abs(key-self.d).sum(1)
        self.reward[mask] = value[mask]
        self.reason[mask] = reason
        self.done[mask] = True

    def _capture(self):
        hit = (self.a == self.d[:, None]).all(2)
        self._finish(hit.any(1), 1, hit[np.arange(self.n), 1-self.role])

    def attacker_phase(self, action, directions=None):
        modes = np.column_stack([np.where(self.role == 0, action//2, action%2),
                                 np.where(self.role == 1, action//2, action%2)])
        if directions is None:
            directions = sample(MODE_PROBS[modes].reshape(-1, 4), self.rng).reshape(-1, 2)
        old = self.a.copy(); live = ~self.done
        self.a[live] = np.clip(old[live]+MOVES[directions[live]], 0, self.size-1)
        destinations = np.clip(old[:, :, None]+MOVES, 0, self.size-1)
        lk = (destinations == self.a[:, :, None]).all(-1) @ MODE_PROBS.T
        self.logodds[live]+=(np.log(lk[live,0,1])+np.log(lk[live,1,0])-
                            np.log(lk[live,0,0])-np.log(lk[live,1,1]))
        self.b[live]=expit(self.logodds[live])
        self._capture()
        self._finish(self.a[np.arange(self.n), self.role, 1] == 0, 2)

    def defender_phase(self, action):
        move = DEFENDER_MOVES[action]; start = self.d.copy()
        for step in (np.sign(move), move):
            live = ~self.done
            self.d[live] = np.clip(start[live]+step[live], 0, self.size-1)
            self._capture()
        if self.t == self.horizon-1: self._finish(~self.done, 3)
        self.t += 1

def teacher(e, aware):
    """Role-blind task teacher for independently pretrained endpoint networks."""
    dest = np.clip(e.d[:, None]+DEFENDER_MOVES, 0, e.size-1)
    distance = np.abs(dest[:, :, None]-e.a[:, None]).sum(-1)
    # Blind pursues nearest attacker; aware pursues expected key-holder distance.
    cost = ((e.b[:, None]*distance[:, :, 0]+(1-e.b[:, None])*distance[:, :, 1])
            if aware else distance.min(-1))
    z = -2.*cost; p = np.exp(z-z.max(1, keepdims=True))
    return p/p.sum(1, keepdims=True)
