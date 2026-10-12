#!/usr/bin/env python3
"""calopt60 core: OOF calibration/threshold transforms + kit03 rescore.

Record tuple: (cid,gid,rid,field,qt,keys,vals,noul,gold,st,winner)
Decision-preservation contract: only rows whose banked winner == raw argmax
may be overridden by a choice transform; rule/K2 rows preserved.
"""
import sys, os, math, json, pickle, collections, hashlib, bisect, random
_LEVERS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_LEVERS, "bundle60"))
import scorekit as K
import metrics_local as M

HERE = os.path.join(_LEVERS, "calopt60")
TABLE = K.TABLE
load_suite_rows = K.load_suite_rows

def fold_of(gid, nf=5):
    return hashlib.md5(gid.encode()).digest()[0] % nf

def argmax_idx(vals):
    return max(range(len(vals)), key=lambda i: vals[i])

def raw_argmax(keys, vals):
    return keys[argmax_idx(list(vals))]

def softmax(xs):
    m = max(xs); es=[math.exp(x-m) for x in xs]; s=sum(es); return [e/s for e in es]

def logit(p):
    p=float(p)
    if p<=0: return -30.0
    if p>=1: return 30.0
    return math.log(p/(1-p))

def sigmoid(x):
    if x<-30: return 0.0
    if x>30: return 1.0
    return 1/(1+math.exp(-x))

# ---------------- metrics per catalog ----------------
def metric_fn(cid):
    if cid == 59:
        return lambda items: M.binary_f1([dict(pred=(x['pred'] is True), gold=(x['gold'] is True)) for x in items], True)
    if cid == 38:
        return M.review_f1
    if cid == 42:
        return lambda items: M.macro_f1([(x['gold'], x['pred']) for x in items])
    if cid in (33, 1):
        return M.case_exact
    return M.accuracy

# ---------------- collect items ----------------
def collect(recs, cid):
    out = []
    for r in recs:
        c, gid, rid, field, qt, keys, vals, noul, gold, st, winner = r
        if c != cid:
            continue
        if qt == 'noul':
            out.append(dict(idx=None, gid=gid, rid=rid, field=field, qt='noul',
                            p=noul, gold=(gold in (1, True, 'yes', 'true', 'True', 'SUPPORTED')),
                            winner=None, keys=None, vals=None, ra=None,
                            pred=(noul >= 0.5)))
        else:
            keys = list(keys); vals = [float(v) for v in vals]
            ra = keys[argmax_idx(vals)]
            out.append(dict(idx=None, gid=gid, rid=rid, field=field, qt='choice',
                            gold=gold, winner=winner, keys=keys, vals=vals, ra=ra,
                            pred=winner))
    return out

# ---------------- PAV ----------------
def pav(xs, ys):
    xs = list(xs)
    if not xs: return [], []
    val=list(ys); w=[1.0]*len(xs); i=0
    while i < len(val)-1:
        if val[i] > val[i+1]:
            tw=w[i]+w[i+1]; tv=(val[i]*w[i]+val[i+1]*w[i+1])/tw
            xs[i:i+2]=[(xs[i]+xs[i+1])/2.0]; val[i:i+2]=[tv]; w[i:i+2]=[tw]
            if i>0: i-=1
        else: i+=1
    return xs, val

def fit_isotonic(pairs):
    s=sorted(pairs)
    return pav([p for p,_ in s],[y for _,y in s])

def apply_iso(m, p):
    xs,ys=m
    if not xs: return 0.5
    if p<=xs[0]: return ys[0]
    if p>=xs[-1]: return ys[-1]
    i=bisect.bisect_left(xs,p); return ys[min(i,len(ys)-1)]

# ---------------- OOF choice transforms ----------------
def oof_choice_threshold(items, pos, metric, nf=5, grid=None, shrink=1.0):
    """Return dict (rid,field)->newchoice for eligible rows.  Does not leave
    caller item['pred'] mutated except through the returned override map."""
    elig = [x for x in items if x['elig']]
    ov = {}
    for f in range(nf):
        tr = [x for x in elig if fold_of(x['gid'], nf) != f]
        te = [x for x in elig if fold_of(x['gid'], nf) == f]
        if not tr:
            continue
        if grid is not None:
            cand = grid
        else:
            ps = sorted(x['p'] for x in tr)
            n = len(ps)
            cand = sorted(set(ps[min(n-1, int(n*i/60))] for i in range(1, 60)))
        best = None; bestv = -1.0
        for t in cand:
            saved = [x['pred'] for x in tr]
            for x in tr:
                x['pred'] = pos if x['p'] > t else x['other']
            v = metric(tr)
            for x, s in zip(tr, saved):
                x['pred'] = s
            if v > bestv + 1e-12:
                bestv = v; best = t
        t = best
        if shrink != 1.0:
            t = 0.5 + shrink * (t - 0.5)
        for x in te:
            npred = pos if x['p'] > t else x['other']
            if npred != x['winner']:
                ov[(x['rid'], x['field'])] = npred
    return ov

# ---------------- OOF noul transforms ----------------
def oof_noul(items, nf=5, mode='iso', nbin=None, tau=None, eps=None, prior=None):
    """Return dict (rid,field)->new noul for noul items."""
    out = {}
    for f in range(nf):
        tr = [x for x in items if fold_of(x['gid'], nf) != f]
        te = [x for x in items if fold_of(x['gid'], nf) == f]
        if not tr: continue
        if mode == 'iso':
            m = fit_isotonic([(x['p'], 1.0 if x['gold'] else 0.0) for x in tr])
            for x in te: out[(x['rid'], x['field'])] = apply_iso(m, x['p'])
        elif mode == 'binned':
            buckets = collections.defaultdict(list)
            for x in tr: buckets[min(nbin-1, int(x['p']*nbin))].append(1.0 if x['gold'] else 0.0)
            bxs = sorted(buckets)
            for x in te:
                b = min(nbin-1, int(x['p']*nbin))
                j = max(0, min(bisect.bisect_right(bxs, b)-1, len(bxs)-1))
                out[(x['rid'], x['field'])] = sum(buckets[bxs[j]])/len(buckets[bxs[j]])
        elif mode == 'platt':
            # logistic on logit(p)
            xs=[logit(x['p']) for x in tr]; ys=[1.0 if x['gold'] else 0.0 for x in tr]
            a,b = 1.0,0.0
            for _ in range(200):
                ga=gb=0.0
                for xx,yy in zip(xs,ys):
                    pp=sigmoid(a*xx+b); ga+=(pp-yy)*xx; gb+=(pp-yy)
                a-=0.1*ga/len(xs); b-=0.1*gb/len(xs)
            for x in te: out[(x['rid'],x['field'])] = sigmoid(a*logit(x['p'])+b)
        elif mode == 'thr':
            for x in te: out[(x['rid'],x['field'])] = 1.0 if x['p'] > tau else 0.0
        elif mode == 'shrink':
            for x in te: out[(x['rid'],x['field'])] = (1-eps)*x['p'] + eps*prior
    return out

# ---------------- build results and rescore ----------------
def build_results(recs, choice_ov=None, noul_ov=None):
    results = {}
    for r in recs:
        cid, gid, rid, field, qt, keys, vals, noul, gold, st, winner = r
        if qt == 'noul':
            v2 = noul_ov[(rid, field)] if (noul_ov and (rid, field) in noul_ov) else noul
            ans = {"type": "noul", "noul": v2}
        else:
            keys = list(keys); vals = [float(v) for v in vals]
            ra = raw_argmax(keys, vals)
            choice = winner
            if choice_ov and (rid, field) in choice_ov and (winner == ra or winner is None):
                choice = choice_ov[(rid, field)]
            ans = {"type": "choice", "choice": choice,
                   "probabilities": {keys[i]: vals[i] for i in range(len(keys))}}
        slot = results.get(rid)
        if slot is None:
            slot = {"run_id": rid, "catalog_id": cid, "status": "ok",
                    "response": {"answers": {}}, "total_wall_ms": 0.0}
            results[rid] = slot
        slot["response"]["answers"][field] = ans
    return results

def rescore(name, recs, edition, rows, choice_ov=None, noul_ov=None):
    res = build_results(recs, choice_ov, noul_ov)
    out = os.path.join(HERE, "score", name)
    os.makedirs(out, exist_ok=True)
    s = K.score(res, edition, rows, out)
    return s["decision_index"], s["raw_index"]
