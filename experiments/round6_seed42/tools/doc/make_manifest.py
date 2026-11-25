"""manifest.json of the recommended file = the export's own manifest + local validation numbers read from the score CSVs."""
import csv, glob, collections, json
R = "/Volumes/Data/Projects/统计学习大作业/repo/experiments/round6_seed42/"
S = R + "scores/"
def load(pat):
    rows = []
    for f in sorted(glob.glob(S + pat)): rows += list(csv.DictReader(open(f)))
    return rows
W = {"heart/deep": .48, "heart/mid": .16, "heart/shallow": .16, "all/asis": .2}
W5 = {"heart/deep": .3, "heart/mid": .1, "heart/shallow": .1, "all/asis": .5}
R5, R6, R5H, R6B, R5B, R6O = "R5 (round-5 file)", "R6 (candidate)", "R5 base, harmonised anchor", "R6 base only", "R5 base only", "R6 at round-5 share"
EIGHT = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all", "raw", "h3", "dm"]
SEVEN = EIGHT[1:]
def block(split):
    sc = collections.defaultdict(dict); px = collections.defaultdict(dict)
    for r in load(f"round6_final__scen_{split}_rho0.4_trend0.csv"): sc[r["config"]][r["scenario"]] = float(r["score"])
    for r in load(f"round6_final__proxy_{split}_rho0.4_trend0_[ab].csv"): px[r["config"]][r["proxy"]] = float(r["score"])
    exp = lambda c, w: sum(w[s] * sc[c][s] for s in w)
    d2 = [px[R5H][p] - px[R5][p] for p in EIGHT]
    r2 = lambda v: round(v, 2)
    return {
        "condition_1a_base_D/raw_R6_minus_R5 (>= +1.0)": r2(px[R6B]["D/raw"] - px[R5B]["D/raw"]),
        "condition_1a_base_D/dm_R6_minus_R5 (>= +1.0)": r2(px[R6B]["D/dm"] - px[R5B]["D/dm"]),
        "condition_1b_gen_R6_minus_R5 (>= -1.0)": r2(px[R6]["gen"] - px[R5]["gen"]),
        "condition_2_harmonised_anchor_on_R5_base_mean_of_8_proxies (>= -1.0)": r2(sum(d2) / 8),
        "condition_2_minimum_of_8_proxies (>= -2.0)": r2(min(d2)),
        "condition_3a_scenario_table_weighted_R6_minus_R5 (>= +1.0)": r2(exp(R6, W) - exp(R5, W)),
        "condition_3b_same_with_heart_probability_0.5 (>= 0)": r2(exp(R6, W5) - exp(R5, W5)),
        "scenario_table": {c: {s: r2(sc[c][s]) for s in W} for c in (R5, R6, R6O)},
        "proxies_R6_minus_R5_not_binding": {p: r2(px[R6][p] - px[R5][p]) for p in SEVEN},
        "proxies_first_version_minus_R5_not_binding": {p: r2(px[R6O][p] - px[R5][p]) for p in SEVEN},
    }
m = json.load(open(R + "manifest_dorm.json"))
m["local_validation"] = {
    "note": "Local calibrated scores (copying a proxy's reference stage = 50, the other half of its target = 100); not leaderboard scores. 'R5' is the round-5 method re-run inside each proxy, 'R6 (candidate)' this configuration in proxy form (anchor share = 0.4 x elapsed share), 'R6 at round-5 share' the first version of the candidate (anchor share = 0.75 x elapsed share, 0.375 at test).",
    "replacement_rule": "conditions written to fable_进展.md at 2026-11-26 02:10 (Asia/Shanghai), before the confirmation splits were first scored (02:13, first version of the candidate, anchor share 0.375). After the first independent review the anchor share was re-chosen on development tables only (0.20) and the rule for this file was written at 03:00, before its confirmation scores were computed: the same three conditions against R5; the choice between 0.20 and 0.375 does not use confirmation data. All conditions met.",
    "caveat": "Conditions 1a and 3 use targets with zero elapsed time taken from the E10.5 heart library, which favours R6 by construction; every local proxy in which time passes and the external E14.5 target (TE) prefer the round-5 method. The recommendation is a judgement resting on the organisers' Data Protocol and on the test-board score of the first submission; see README.",
    "development": block("development"),
    "confirmation": block("confirmation"),
    "TE_confirmation_base_only (not binding)": {"library_3_natural": {"te/raw": 61.28, "te/dm": 60.85}, "round5_base": {"te/raw": 66.1, "te/dm": 65.09}},
    "anchor_share_choice_development_only": {
        "families": "A = mean of the four proxies targeting E10.5 (p8/deep, p8/all, 2d/deep, 2d/all); B = gen; C = zero-elapsed-time scenarios (deep/mid/shallow weighted 0.6/0.2/0.2); oracle base, harmonised anchor, share = rho x elapsed share, linear interpolation between grid points",
        "rho": [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
        "A": [65.59, 68.86, 70.40, 71.69, 72.80, 73.60, 74.59],
        "B": [59.39, 60.84, 61.56, 61.41, 61.26, 61.12, 60.03],
        "C": [95.13, 92.48, 91.15, 88.76, 85.17, 82.04, 76.78],
        "max_regret_over_three_weightings": {"0.3": 0.88, "0.4": 0.41, "0.5": 1.29, "0.6": 2.12, "0.75": 3.91},
        "chosen_rho": 0.4},
    "score_tables": "experiments/round6_seed42/scores/",
}
m["code_paths"] = {"build": "experiments/round6_config.py::build", "draw_and_mix": "experiments/round6_method.py::predict",
                   "harmonisation": "experiments/round6_anchor.py::common_shift, harmonise", "anchor_reader": "experiments/round5_anchor.py::load_anchor",
                   "export": "experiments/round6_export.py", "evaluation": "experiments/round6_final.py, experiments/round6_suite.py",
                   "public": False, "note": "code is not public; available on request"}
m["fallback"] = {"file": "round5_anchor0.375_all_h1", "sha256": "7855b636c13b7de3661f3588096cdaabebe34bd088a15599dce06251e7997ee0", "where": "第二次提交_round5_anchor/"}
m["first_version_not_recommended"] = {"file": "round6_heartlib_anchor0.375_harm1_trend0", "sha256": "8c86cedc2903f3e4705ebe5004dd6c4ce3deecc853cf47bf0819338252d16778",
                                      "manifest": "manifest_anchor0.375_not_recommended.json"}
m["ai_involvement"] = "The method was developed and run with AI coding agents (Claude Code) under the team's direction; the team chose which file to submit."
json.dump(m, open(R + "manifest.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(m["local_validation"]["development"], ensure_ascii=False, indent=1)[:1500])
print(json.dumps(m["local_validation"]["confirmation"], ensure_ascii=False, indent=1)[:2500])
