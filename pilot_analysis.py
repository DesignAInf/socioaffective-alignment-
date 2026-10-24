#!/usr/bin/env python3
"""
Pilot application of the restitution codes (a, f, s) to psychosis-bench transcripts.
Companion to: "Socioaffective Alignment, One Relationship at a Time", Section 4.3.

Steps
  1. parse   - parse the public psychosis-bench batch log into turn-level records
  2. sample  - fixed sample: turns 4, 8 and 12 of all 128 conversations (n = 384)
  3. code    - code each AI turn for a / f (hedged) / s units with a language model
               (OPTIONAL: requires an API; the coded file used in the paper is provided)
  4. analyse - pre-specified analyses (see analysis_plan.md); writes results_table.csv
  5. figure  - Figure 3 of the paper (fig_demo.pdf / fig_demo.png)

Usage
  python pilot_analysis.py analyse --coded psychosisbench_coded_afs.csv
  python pilot_analysis.py figure  --coded psychosisbench_coded_afs.csv
  python pilot_analysis.py parse   --log 20250905_000927_async_batch_8models_16cases.md

Data source: https://github.com/w-is-h/psychosis-bench (MIT licence),
file outputs/20250905_000927_async_batch_8models_16cases.md
Requirements: python>=3.10, pandas, numpy, scipy, statsmodels>=0.14, matplotlib
"""
import argparse, json, re, sys
import numpy as np
import pandas as pd

LOG_URL = ("https://raw.githubusercontent.com/w-is-h/psychosis-bench/main/outputs/"
           "20250905_000927_async_batch_8models_16cases.md")
SAMPLE_TURNS = (4, 8, 12)
FORMS = ["Marking only\n(a, no f)", "Marking with endorsement\n(a and f)",
         "Endorsement only\n(f, no a)", "Neither a nor f"]

# ---------------------------------------------------------------- 1. parse
def parse_log(text):
    exps = re.split(r"\n## Experiment (\d+): ", text)
    recs = []
    for k in range(1, len(exps), 2):
        eid, body = int(exps[k]), exps[k + 1]
        meta = {f: re.search(rf"\*\*{f}:\*\* (.+)", body).group(1).strip()
                for f in ["Model", "Test Case", "Theme", "Condition"]}
        turns = re.split(r"\n### Turn (\d+)\n", body)
        users = []
        for t in range(1, len(turns), 2):
            tn, tb = int(turns[t]), turns[t + 1]
            u = re.search(r"\*\*👤 User:\*\*\n(.*?)\n\n\*\*🤖 Assistant:\*\*", tb, re.S)
            a = re.search(r"\*\*🤖 Assistant:\*\*\n(.*?)(?:\n\n\*\*📊 Scores:\*\*|\n---|\Z)", tb, re.S)
            dcs = re.search(r"DCS \(Delusion Confirmation Score\): (-?\d)", tb)
            ut = u.group(1).strip() if u else ""
            recs.append(dict(exp=eid, model=meta["Model"], case=meta["Test Case"],
                             theme=meta["Theme"], cond=meta["Condition"], turn=tn,
                             user=ut, prior_users=list(users),
                             assistant=a.group(1).strip() if a else "",
                             dcs=int(dcs.group(1)) if dcs else None))
            users.append(ut)
    return pd.DataFrame(recs)

def sample(df):
    return df[df.turn.isin(SAMPLE_TURNS)].reset_index(drop=True)

# ---------------------------------------------------------------- 3. code (optional)
def build_request(row, maxchars=9000):
    """Return (system, user) prompts. The system prompt is coder_prompt.txt (verbatim, as in the pilot)."""
    system = open("coder_prompt.txt").read()
    ctx = "\n".join(f"USER turn {i+1}: {u}" for i, u in enumerate(row.prior_users))
    user = (f"USER TURNS SO FAR (the user's frame comes only from these):\n{ctx}\n"
            f"USER turn {row.turn} (current): {row.user}\n\n"
            f"AI TURN TO CODE:\n<<<\n{row.assistant[:maxchars]}\n>>>\n\n"
            "Code every restitution unit in the AI turn with the tool.")
    return system, user

TOOL_SCHEMA = {"name": "code_turn",
    "input_schema": {"type": "object", "properties": {
        "a_units": {"type": "array", "items": {"type": "string"}},
        "f_units": {"type": "array", "items": {"type": "object", "properties": {
            "quote": {"type": "string"}, "hedged": {"type": "boolean"},
            "user_source": {"type": "string"}}, "required": ["quote", "hedged", "user_source"]}},
        "s_units": {"type": "array", "items": {"type": "string"}}},
        "required": ["a_units", "f_units", "s_units"]}}

def code_turn(system, user):
    """Connect to a language-model API that supports forced tool use with TOOL_SCHEMA and
    return the tool input as a dict. In the pilot, the coder was a model from a different
    family than the DCS judge (o4-mini), with thinking disabled. Not implemented here."""
    raise NotImplementedError("Connect code_turn() to your model API to re-run the coding step.")

# ---------------------------------------------------------------- 4. analyse
def load_coded(path):
    d = pd.read_csv(path)
    d["dcs"] = d["dcs"].astype(int)
    for c in "afs":
        d[c] = (d[c + "_n"] > 0).astype(int)
    d["n_units"] = d.a_n + d.f_n + d.s_n
    d["M"] = np.where(d.n_units > 0, d.a_n / d.n_units.replace(0, np.nan), np.nan)
    d["form"] = np.select([(d.a == 1) & (d.f == 0), (d.a == 1) & (d.f == 1), (d.a == 0) & (d.f == 1)],
                          FORMS[:3], FORMS[3])
    return d

def fit_ordinal(dd, preds, label):
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    X = pd.concat([dd[preds].astype(float),
                   pd.get_dummies(dd[["model", "case"]], drop_first=True).astype(float)], axis=1)
    res = OrderedModel(dd.dcs, X, distr="logit").fit(
        method="bfgs", disp=False, maxiter=2000, cov_type="cluster", cov_kwds={"groups": dd.exp.values})
    return pd.DataFrame([dict(analysis=label, term=p, OR=np.exp(res.params[p]),
                              lo=np.exp(res.params[p] - 1.96 * res.bse[p]),
                              hi=np.exp(res.params[p] + 1.96 * res.bse[p]),
                              p=res.pvalues[p], n=len(dd)) for p in preds])

def analyse(d):
    import statsmodels.formula.api as smf
    from scipy.stats import spearmanr
    rho, p = spearmanr(d.a_n, d.f_n)
    print(f"N = {len(d)}; presence a/f/s = {d.a.mean():.3f}/{d.f.mean():.3f}/{d.s.mean():.3f}")
    print(f"share with a and f = {(d.a & d.f).mean():.3f}; Spearman(a_n, f_n) = {rho:.3f} (p = {p:.2g})")
    print("DCS distribution by form:\n", pd.crosstab(d.form, d.dcs, normalize="index").round(3))
    dM = d[d.n_units > 0]
    tabs = [fit_ordinal(d, ["a", "f", "s"], "A2/A3 primary: ordinal, a+f+s"),
            fit_ordinal(dM, ["M"], "A4: ordinal, M alone (eligible turns)"),
            fit_ordinal(dM, ["M", "f"], "A4: ordinal, M + f"),
            fit_ordinal(d[d.model != "anthropic/claude-sonnet-4"], ["a", "f", "s"], "A5: excl. claude-sonnet-4")]
    d = d.assign(dcs2=(d.dcs == 2).astype(int))
    lg = smf.logit("dcs2 ~ a + f + s + C(model) + C(case)", data=d).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": d.exp.values})
    tabs.append(pd.DataFrame([dict(analysis="A5: logistic DCS=2", term=t, OR=np.exp(lg.params[t]),
                                   lo=np.exp(lg.params[t] - 1.96 * lg.bse[t]),
                                   hi=np.exp(lg.params[t] + 1.96 * lg.bse[t]),
                                   p=lg.pvalues[t], n=len(d)) for t in ["a", "f", "s"]]))
    out = pd.concat(tabs, ignore_index=True)
    print(out.round(3).to_string(index=False))
    return out

# ---------------------------------------------------------------- 5. figure
def figure(d, rt, stem="fig_demo"):
    import matplotlib as mpl, matplotlib.pyplot as plt
    tab = pd.crosstab(d.form, d.dcs, normalize="index").reindex(FORMS)
    ns = d.form.value_counts().reindex(FORMS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.75})
    cols = ["#dbe6f1", "#8fb3d6", "#2f6aa3"]; y = np.arange(len(FORMS))[::-1]; left = np.zeros(len(FORMS))
    for k, c in zip([0, 1, 2], cols):
        v = tab[k].values; ax1.barh(y, v, left=left, color=c, edgecolor="white", height=0.62, label=f"DCS {k}"); left += v
    ax1.set_yticks(y); ax1.set_yticklabels([f"{o}\nn = {ns[o]}" for o in FORMS]); ax1.set_xlim(0, 1)
    ax1.set_xlabel("Share of AI turns"); ax1.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax1.set_title("A  DCS by restitution form", loc="left")
    ax1.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08))
    rows = [("A2/A3 primary: ordinal, a+f+s", "a", "Marking (a)"), ("A2/A3 primary: ordinal, a+f+s", "f", "Endorsement (f)"),
            ("A2/A3 primary: ordinal, a+f+s", "s", "AI's own affect (s)"),
            ("A4: ordinal, M alone (eligible turns)", "M", "M alone"), ("A4: ordinal, M + f", "M", "M, adjusted for f")]
    yy = np.arange(len(rows))[::-1]
    for yv, (an, tm, lab) in zip(yy, rows):
        r = rt[(rt.analysis == an) & (rt.term == tm)].iloc[0]; foc = tm == "f"
        ax2.plot([r.lo, r.hi], [yv, yv], color="#2f6aa3" if foc else "0.45", lw=1.4 if foc else 1.0)
        ax2.plot(r.OR, yv, "o", color="#2f6aa3" if foc else "0.35", ms=5 if foc else 4)
    ax2.axvline(1, color="0.6", lw=0.8, ls="--"); ax2.axhline(1.5, color="0.85", lw=0.6); ax2.set_xscale("log")
    ax2.set_xticks([0.01, 0.1, 1, 10]); ax2.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax2.set_yticks(yy); ax2.set_yticklabels([r[2] for r in rows]); ax2.set_xlabel("Odds ratio for higher DCS (95% CI)")
    ax2.set_title("B  Ordinal models (clustered 95% CI)", loc="left")
    fig.savefig(stem + ".pdf", bbox_inches="tight"); fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["parse", "analyse", "figure"])
    ap.add_argument("--log"); ap.add_argument("--coded", default="psychosisbench_coded_afs.csv")
    ap.add_argument("--out", default="results_table.csv"); a = ap.parse_args()
    if a.step == "parse":
        text = open(a.log).read() if a.log else __import__("urllib.request").request.urlopen(LOG_URL).read().decode()
        df = parse_log(text); s = sample(df)
        print(df.shape, "turns;", len(s), "sampled;", s.dcs.notna().sum(), "with DCS")
        s.to_json("sample_turns.json", orient="records", force_ascii=False)
    elif a.step == "analyse":
        analyse(load_coded(a.coded)).to_csv(a.out, index=False)
    elif a.step == "figure":
        figure(load_coded(a.coded), pd.read_csv(a.out))

if __name__ == "__main__":
    main()
