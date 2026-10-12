#!/usr/bin/env python3
"""Local replicas of kit03 native metrics for single catalogs, so we can sweep
calibration/threshold transforms cheaply (validated against score/identity).
Records: (cid,gid,rid,field,qt,keys,vals,noul,gold,st,winner)"""
import collections

def _yes(g):
    return g in (1, True, "yes", "true", "True", "SUPPORTED")

def macro_f1(pairs):
    labels = set(x for p in pairs for x in p)
    res = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in pairs)
        fp = sum(g != lab and p == lab for g, p in pairs)
        fn = sum(g == lab and p != lab for g, p in pairs)
        res[lab] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return sum(res.values()) / len(res) if res else 0.0

def binary_f1(items, positive):
    tp = sum(x['pred'] == positive and x['gold'] == positive for x in items)
    fp = sum(x['pred'] == positive and x['gold'] != positive for x in items)
    fn = sum(x['pred'] != positive and x['gold'] == positive for x in items)
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0

def review_f1(items):
    groups = collections.defaultdict(list)
    for x in items:
        groups[x['gid']].append(x)
    vals = []
    for g, xs in groups.items():
        gold_yes = set((x['rid'], x['field']) for x in xs if x['gold'] in ('yes', True))
        pred_yes = set((x['rid'], x['field']) for x in xs if x['pred'] in ('yes', True))
        d = len(gold_yes) + len(pred_yes)
        vals.append(2 * len(gold_yes & pred_yes) / d if d else 1.0)
    return sum(vals) / len(vals) if vals else 0.0

def accuracy(items):
    return sum(x['pred'] == x['gold'] for x in items) / len(items) if items else 0.0

def case_exact(items):
    groups = collections.defaultdict(list)
    for x in items:
        groups[x['gid']].append(x)
    return sum(all(x['pred'] == x['gold'] for x in xs) for xs in groups.values()) / len(groups) if groups else 0.0

def collect(recs, cid):
    items = []
    for r in recs:
        c, gid, rid, field, qt, keys, vals, noul, gold, st, winner = r
        if c != cid:
            continue
        if qt == 'noul':
            items.append(dict(gid=gid, rid=rid, field=field, p=noul, gold=_yes(gold), pred=None))
        else:
            items.append(dict(gid=gid, rid=rid, field=field, keys=list(keys),
                              vals=list(vals), gold=gold, pred=winner))
    return items
