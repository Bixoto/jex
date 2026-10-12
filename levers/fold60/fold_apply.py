#!/usr/bin/env python3
"""Apply the cat62 positional fold post-hoc to an engine results capture.

Generic form of fold_cat62.py: --in, --out, --name. Same rule/semantics.
Writes a receipt and a folded results.jsonl.gz.
"""
import gzip, json, math, collections, os, argparse, sys

PHRASE = "Which response should the assistant give to the user's question, given the available tools?"
POSITIVE = "D"
TAU = -13.63
TARGET_SIG = "choice|A,B,C,D"

def logit(p):
    p=float(p)
    if p<=0.0: return float("-inf")
    if p>=1.0: return float("inf")
    return math.log(p)-math.log1p(-p)

def argmax(probs,keys): return max(keys,key=lambda k:probs[k])
def sig_of(keys): return "choice|"+",".join(sorted({str(k) for k in keys}))
def apply_rule(probs,keys,positive,tau):
    if logit(probs[positive])>=float(tau): return positive
    others=[k for k in keys if k!=positive]
    if not others: return positive
    return max(others,key=lambda k:probs[k])
def inv_tbo(emitted,keys,T):
    w={k:(max(float(emitted.get(k,0.0)),0.0))**float(T) for k in keys}
    s=sum(w.values())
    return {k:(v/s if s>0 else v) for k,v in w.items()}

def fold(path_in, path_out):
    counters=collections.Counter(); per_cat=collections.defaultdict(collections.Counter)
    changed_rows=[]; mismatch=0; n_rows=0
    with gzip.open(path_in,"rt",encoding="utf-8") as fin, \
         gzip.open(path_out,"wt",encoding="utf-8") as fout:
        for line in fin:
            r=json.loads(line); n_rows+=1
            cid=r.get("catalog_id")
            for qk,q in (r.get("payload",{}).get("questions",{}) or {}).items():
                if q.get("type")!="choice": continue
                keys=list((q.get("criteria") or {}).keys())
                if sig_of(keys)!=TARGET_SIG: continue
                if PHRASE.lower() not in str(q.get("instructions") or "").lower(): continue
                ans=r["response"]["answers"].get(qk)
                if ans is None or ans.get("type")!="choice": continue
                emitted=ans.get("probabilities",{})
                T=r.get("raw_output",{}).get(qk,{}).get("emitted_temperature")
                work=inv_tbo(emitted,keys,T)
                am=argmax(emitted,keys)
                rc=apply_rule(work,keys,POSITIVE,TAU)
                counters["matched"]+=1; per_cat[cid]["matched"]+=1
                if am!=ans.get("choice"): mismatch+=1
                if rc!=am:
                    counters["changed"]+=1; per_cat[cid]["changed"]+=1
                    if rc==POSITIVE: counters["changed_to_D"]+=1
                    changed_rows.append((r["run_id"],ans.get("choice"),rc))
                    ans["choice"]=rc
            fout.write(json.dumps(r,ensure_ascii=False)+"\n")
    return {
        "source":path_in,"output":path_out,"rows":n_rows,
        "rule":{"signature":TARGET_SIG,"positive":POSITIVE,"tau_logodds":TAU,"instructions_contains":PHRASE},
        "counters":dict(counters),
        "per_catalog":{str(c):dict(v) for c,v in per_cat.items()},
        "matched_catalogs":sorted({int(c) for c in per_cat}),
        "stored_choice_vs_argmax_mismatch_on_matched":mismatch,
        "changed_from":dict(collections.Counter(f for _,f,_ in changed_rows)),
        "changed_to":dict(collections.Counter(t for _,_,t in changed_rows)),
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="inp",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--name",required=True)
    ap.add_argument("--receipt",default=None)
    a=ap.parse_args()
    rec=fold(a.inp,a.out)
    rp=a.receipt or (a.out+".receipt.json")
    with open(rp,"w") as f: json.dump(rec,f,indent=2)
    print(a.name, json.dumps(rec["counters"]), "cats", rec["matched_catalogs"],
          "from",rec["changed_from"],"to",rec["changed_to"])
