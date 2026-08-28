#!/usr/bin/env python3
# ablations.py — reproduce the paper §5.4 ablation numbers with bench.py (10 seeds).
import os, math, time
import bench as B

SEEDS = list(range(1, 11))
H_PRED = 24000
H_POL = 20000

def preds(pairs):
    # pairs: list of (label, predictor_name); returns dict label -> [catches]*10
    out = {}
    for label, name in pairs:
        out[label] = [B.run_episode('circling', name, s, H_PRED, True)['catches'] for s in SEEDS]
    return out

def pols(pairs):
    out = {}
    for label, name in pairs:
        out[label] = [B.run_policy_episode('circling', name, s, H_POL, True)['catches'] for s in SEEDS]
    return out

def line(d):
    return ', '.join(f'{k}: {B._mean_sd(v)[0]:.0f}±{B._mean_sd(v)[1]:.0f}' for k, v in d.items())

t0 = time.time()

# 1. turn-rate clamp
print('=== 1. turn-rate clamp (circling, bdh vs velocity-lead) ===')
for T in [1.8, 2.7, 3.6, 5.4]:
    B.CHASER_MAXTURN = T
    d = preds([('vel', 'velocity-lead'), ('bdh', 'bdh')])
    print(f'TURN={T}: ' + line(d))
B.CHASER_MAXTURN = 3.6

# 2. speed ratio
print('=== 2. speed ratio (circling, bdh vs velocity-lead vs accel-lead) ===')
for sp in [150, 165, 180]:
    B.PREY_SPEED = sp
    d = preds([('vel', 'velocity-lead'), ('accel', 'accel-lead'), ('bdh', 'bdh')])
    print(f'PREY_SPEED={sp}: ' + line(d))
B.PREY_SPEED = 165

# 3. memory width (Fourier features M)
print('=== 3. memory width M (circling, bdh) ===')
for M in [0, 8, 24, 48]:
    os.environ['BDH_M'] = str(M)
    d = preds([('bdh', 'bdh')])
    print(f'M={M}: ' + line(d))
os.environ['BDH_M'] = '0'

# 4. learning rate eta
print('=== 4. learning rate eta (circling, bdh) ===')
for e in [0.1, 0.3, 0.5, 1.0]:
    os.environ['BDH_ETA'] = str(e)
    d = preds([('bdh', 'bdh')])
    print(f'eta={e}: ' + line(d))
os.environ['BDH_ETA'] = '0.5'

# 5. weight decay lam
print('=== 5. weight decay lam (circling, bdh) ===')
for L in [0.0, 1e-4, 1e-3, 1e-2]:
    os.environ['BDH_LAM'] = str(L)
    d = preds([('bdh', 'bdh')])
    print(f'lam={L}: ' + line(d))
os.environ['BDH_LAM'] = '1e-3'

# 6. action discretization (policies)
print('=== 6. action discretization NACT (circling, policies) ===')
for N in [4, 8, 16]:
    B.N_ACTIONS = N
    B.ACTIONS = [a * (2 * math.pi / N) for a in range(N)]
    d = pols([('reflex', 'pure-pursuit'), ('linq', 'linear-q'), ('dqn', 'dqn'), ('ppo', 'ppo')])
    print(f'NACT={N}: ' + line(d))
B.N_ACTIONS = 8
B.ACTIONS = [a * (2 * math.pi / 8) for a in range(8)]

print(f'\nelapsed {time.time()-t0:.1f}s')
