#!/usr/bin/env python3
"""cat59 (RAGTruth noul) calibration refinement: OOF isotonic/Platt/binned/
threshold/shrinkage + kit rescore + refit-bootstrap CI."""
import sys, os, json, pickle, time, random, collections, bisect, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calopt60"))
import calib_core as C
import metrics_local as M

HERE = C.HERE
COEF = 5.7017  # dindex per unit F1 for cat59 (validated vs kit)

def f1_of(items):
    tp = sum(x['pred'] and x['gold'] for x in items)
    fp = sum(x['pred'] and not x['gold'] for x in items)
    fn = sum((not x['pred']) and x['gold'] for x in items)
    d = 2*tp+fp+fn
    return 2*tp/d if d else 0.0

def fit_mode(tr, mode, lam=1.0, nbin=10, B=0, rng=None):
    ps = [(x['p'], 1.0 if x['gold'] else 0.0) for x in tr]
    if mode == 'iso':
        m = C.fit_isotonic(ps)
        return lambda p: C.apply_iso(m, p)
    if mode == 'iso_shrink':
        m = C.fit_isotonic(ps)
        return lambda p: lam*C.apply_iso(m, p) + (1-lam)*p
    if mode == 'iso_bag':
        maps = []
        for _ in range(B):
            samp = [ps[rng.randrange(len(ps))] for _ in range(len(ps))] if ps else []
            maps.append(C.fit_isotonic(samp))
        def f(p):
            return sum(C.apply_iso(m, p) for m in maps)/len(maps)
        return f
    if mode == 'binned':
        buckets = collections.defaultdict(list)
        for p, y in ps:
            buckets[min(nbin-1, int(p*nbin))].append(y)
        bxs = sorted(buckets)
        def f(p):
            b = min(nbin-1, int(p*nbin))
            j = max(0, min(bisect.bisect_right(bxs, b)-1, len(bxs)-1))
            return sum(buckets[bxs[j]])/len(buckets[bxs[j]])
        return f
    if mode == 'platt':
        xs=[C.logit(p) for p,_ in ps]; ys=[y for _,y in ps]
        a,b=1.0,0.0
        for _ in range(300):
            ga=gb=0.0
            for xx,yy in zip(xs,ys):
                pp=C.sigmoid(a*xx+b); ga+=(pp-yy)*xx; gb+=(pp-yy)
            a-=0.2*ga/len(xs); b-=0.2*gb/len(xs)
        return lambda p: C.sigmoid(a*C.logit(p)+b)
    raise ValueError(mode)

def oof_fit_predict(items, mode, nf=5, seed=0, **kw):
    rng = random.Random(seed)
    pred = {}
    for f in range(nf):
        tr = [x for x in items if C.fold_of(x['gid'], nf) != f]
        te = [x for x in items if C.fold_of(x['gid'], nf) == f]
        if not tr: continue
        fmap = fit_mode(tr, mode, rng=rng, **kw)
        for x in te:
            pred[(x['rid'], x['field'])] = 1.0 if fmap(x['p']) >= 0.5 else 0.0
    return pred

def apply_pred(items, pred):
    for x in items:
        x['pred'] = bool(pred[(x['rid'], x['field'])]) if (x['rid'],x['field']) in pred else (x['p']>=0.5)

def refit_ci(items, mode, nf, B2=1000, seed=123, **kw):
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for x in items: groups[x['gid']].append(x)
    gids = list(groups)
    base = f1_of_with_pred(items)  # base at >=0.5
    dBs=[]
    for _ in range(B2):
        D=[]
        for _ in range(len(gids)):
            D.extend(groups[gids[rng.randrange(len(gids))]])
        # OOF by fold on resample
        pr={}
        rng2=random.Random(rng.randrange(1<<30))
        for f in range(nf):
            tr=[x for x in D if C.fold_of(x['gid'],nf)!=f]
            te=[x for x in D if C.fold_of(x['gid'],nf)==f]
            if not tr: continue
            fm=fit_mode(tr, mode, rng=rng2, **kw)
            for x in te: pr[(x['rid'],x['field'])]=fm(x['p'])
        bp=[(x['p']>=0.5, x['gold']) for x in D]
        ip=[(pr.get((x['rid'],x['field']), x['p'])>=0.5, x['gold']) for x in D]
        dBs.append(COEF*(f1_pairs(ip)-f1_pairs(bp)))
    dBs.sort()
    return dBs[int(.025*len(dBs))], dBs[int(.975*len(dBs))], sum(dBs)/len(dBs)

def f1_pairs(pairs):
    tp=sum(p and g for p,g in pairs); fp=sum(p and not g for p,g in pairs); fn=sum((not p) and g for p,g in pairs)
    d=2*tp+fp+fn; return 2*tp/d if d else 0.0

def f1_of_with_pred(items):
    return f1_pairs([(x['pred'], x['gold']) for x in items])

def main():
    recs = pickle.load(open(C.TABLE, 'rb'))
    edition, rows = C.load_suite_rows()
    items = C.collect(recs, 59)
    for x in items: x['pred'] = x['p'] >= 0.5
    base_f1 = f1_of(items)
    print("cat59 base F1", round(base_f1,5), "n", len(items), flush=True)
    variants = {}
    specs = [
        ("iso_nf5", dict(mode='iso', nf=5)),
        ("iso_nf10", dict(mode='iso', nf=10)),
        ("iso_nf20", dict(mode='iso', nf=20)),
        ("iso_shrink.5_nf5", dict(mode='iso_shrink', nf=5, lam=0.5)),
        ("iso_shrink.75_nf5", dict(mode='iso_shrink', nf=5, lam=0.75)),
        ("iso_shrink.5_nf10", dict(mode='iso_shrink', nf=10, lam=0.5)),
        ("iso_shrink.75_nf10", dict(mode='iso_shrink', nf=10, lam=0.75)),
        ("iso_bag20_nf5", dict(mode='iso_bag', nf=5, B=20)),
        ("iso_bag20_nf10", dict(mode='iso_bag', nf=10, B=20)),
        ("binned10_nf5", dict(mode='binned', nf=5, nbin=10)),
        ("binned20_nf5", dict(mode='binned', nf=5, nbin=20)),
        ("binned20_nf10", dict(mode='binned', nf=10, nbin=20)),
        ("platt_nf5", dict(mode='platt', nf=5)),
        ("platt_nf10", dict(mode='platt', nf=10)),
    ]
    t0=time.time()
    for name, sp in specs:
        pr = oof_fit_predict(items, sp['mode'], nf=sp['nf'], seed=7, **{k:v for k,v in sp.items() if k not in ('mode','nf')})
        apply_pred(items, pr)
        v = f1_of(items)
        noul_ov = {k:(1.0 if p>=0.5 else 0.0) for k,p in pr.items()}
        idx, raw = C.rescore(f"c59_{name}", recs, edition, rows, noul_ov=noul_ov)
        rec = {"f1": round(v,5), "df1": round(v-base_f1,5), "index": idx,
               "dindex": round(idx-59.48,4), "analytic_dindex": round(COEF*(v-base_f1),4),
               "n_flip": sum(1 for x in items if (x['p']>=0.5)!=(x['pred']))}
        variants[name]=rec
        print(name, rec, flush=True)
    json.dump({"base_f1":round(base_f1,5),"COEF":COEF,"variants":variants},
              open(f"{HERE}/receipts/noul_variants_noci.json","w"), indent=1)
    print("point elapsed",round(time.time()-t0,1),flush=True)

    # refit CIs for selected variants
    ci_specs = ["iso_nf5","iso_nf10","iso_nf20","iso_shrink.5_nf10","iso_shrink.75_nf10",
                "iso_bag20_nf10","binned20_nf10","platt_nf10"]
    for name in ci_specs:
        sp = dict(specs)[name] if name in dict(specs) else None
        sp = next(s for n,s in specs if n==name)
        kw = {k:v for k,v in sp.items() if k not in ('mode','nf')}
        lo,hi,m = refit_ci(items, sp['mode'], sp['nf'], B2=1000, seed=99, **kw)
        variants[name]["refit_ci"]=[round(lo,4),round(hi,4)]
        variants[name]["refit_mean"]=round(m,4)
        variants[name]["excludes0"]= lo>0
        print("CI",name,variants[name]["refit_ci"],"mean",variants[name]["refit_mean"],flush=True)
    json.dump({"base_f1":round(base_f1,5),"COEF":COEF,"variants":variants},
              open(f"{HERE}/receipts/noul_variants.json","w"), indent=1)

if __name__ == "__main__":
    main()
