"""Evaluate the keyed sponsor watermark on stored text atlases.

    python scripts/watermark_eval.py --watermarked DIR [DIR ...] --null DIR [DIR ...] \
        [--alpha 0.01] [--output results/watermark_eval.json]

Uses only token IDs and the (public demo) keys, as a detector would. Reports:

* detection: for each watermarked atlas and role (platform, A, B), the share of
  lambda in (0, 1] whose answer is detected at ``alpha``, next to the share of lambda
  where that role drew at least one token (from the recorded ``key`` per step);
* false alarms: the same detector on unwatermarked atlases (real model text from
  the reference or span-hold sampler), as a share of lambda per role;
* robustness: for the answers at lambda = 0.1, 0.5 and 0.9 of each watermarked
  atlas, detection after replacing or deleting a fraction of tokens at random
  (uniform random token IDs) and after truncating to the first n tokens.

Leaves of one atlas share prefixes, so they are not independent samples; shares
are lambda-measures per atlas, averaged over atlases. Synthetic edits are not
paraphrases; nothing here measures robustness to rewriting by a person or a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_auction_lab import watermark as wm  # noqa: E402

PROBE_LAMBDAS = (0.1, 0.5, 0.9)
EDITS = (("substitute", 0.1), ("substitute", 0.25), ("delete", 0.1), ("delete", 0.25),
         ("truncate", 48), ("truncate", 24))
VOCAB = 151_669  # Qwen3 tokenizer size; random substitutions draw ordinary token IDs


def atlases(paths):
    for raw in paths:
        path = Path(raw)
        for file in [path] if path.is_file() else sorted(path.rglob("*.json")):
            try:
                data = json.loads(file.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("schema") == "bid-atlas-v1" \
                    and data.get("kind") == "text":
                yield f"{file.parent.name}/{file.name}", data


def leaf_at(atlas, lam):
    return next(leaf for leaf in atlas["leaves"] if leaf["lo"] < lam <= leaf["hi"])


def edit(token_ids, kind, amount, rng):
    ids = list(token_ids)
    if kind == "truncate":
        return ids[: int(amount)]
    keep = rng.random(len(ids)) >= amount
    if kind == "delete":
        return [t for t, k in zip(ids, keep) if k]
    return [t if k else int(rng.integers(VOCAB)) for t, k in zip(ids, keep)]


def detected(token_ids, keys, alpha):
    found = wm.detect(token_ids, keys)
    return {role: found[role]["p_value"] < alpha for role in wm.ROLES}, found


def evaluate(watermarked, null, alpha=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows, nulls, probes = [], [], []
    for name, atlas in watermarked:
        keys = wm.keys_for_manifest(atlas["manifest"], atlas["bidders"])
        share = {role: 0.0 for role in wm.ROLES}
        drew = {role: 0.0 for role in wm.ROLES}
        for leaf in atlas["leaves"]:
            width = leaf["hi"] - leaf["lo"]
            flags, _ = detected(leaf["token_ids"], keys, alpha)
            roles = {step.get("key") for step in leaf["steps"]}
            for role in wm.ROLES:
                share[role] += width * flags[role]
                drew[role] += width * (role in roles)
        rows.append({"atlas": name, "seed": atlas["seed"], "span_hold": "span_hold" in
                     atlas["manifest"], "detected_share": share, "drew_share": drew})
        for lam in PROBE_LAMBDAS:
            leaf = leaf_at(atlas, lam)
            counts = {role: sum(s.get("key") == role for s in leaf["steps"]) for role in wm.ROLES}
            winner = "A" if counts["A"] >= counts["B"] else "B"
            result = {"atlas": name, "lambda": lam, "tokens_by_role": counts, "winner": winner,
                      "clean": detected(leaf["token_ids"], keys, alpha)[0][winner]}
            for kind, amount in EDITS:
                edited = edit(leaf["token_ids"], kind, amount, rng)
                result[f"{kind}:{amount}"] = detected(edited, keys, alpha)[0][winner]
            probes.append(result)
    for name, atlas in null:
        keys = wm.demo_keys(atlas["bidders"])
        share = {role: 0.0 for role in wm.ROLES}
        for leaf in atlas["leaves"]:
            flags, _ = detected(leaf["token_ids"], keys, alpha)
            for role in wm.ROLES:
                share[role] += (leaf["hi"] - leaf["lo"]) * flags[role]
        nulls.append({"atlas": name, "false_alarm_share": share})
    # Ground truth diagnostic: mean score of tokens by the role that drew them.
    by_truth = {drawn: {role: [] for role in wm.ROLES} for drawn in wm.ROLES}
    for _, atlas in watermarked:
        keys = wm.keys_for_manifest(atlas["manifest"], atlas["bidders"])
        seen = set()
        for leaf in atlas["leaves"]:
            ids = leaf["token_ids"]
            for t, step in enumerate(leaf["steps"]):
                pair = (tuple(ids[max(0, t - wm.CONTEXT_TOKENS):t]), ids[t])
                if pair in seen or step.get("key") not in wm.ROLES:
                    continue
                seen.add(pair)
                for role in wm.ROLES:
                    r = wm.keyed_uniform(keys.key(role), pair[0], pair[1])
                    by_truth[step["key"]][role].append(-np.log1p(-r))
    truth_table = {drawn: {role: {"tokens": len(v), "mean_score": float(np.mean(v)) if v else None}
                           for role, v in row.items()} for drawn, row in by_truth.items()}
    # Campaign-level audit: pool the answers at one lambda across the atlases (seeds) of a
    # scenario and sampler; keys are per scenario (demo keys derive from the bidders).
    groups: dict[str, list] = {}
    for _, atlas in watermarked:
        label = f"{atlas['manifest'].get('scenario')}|span_hold={'span_hold' in atlas['manifest']}"
        groups.setdefault(label, []).append(atlas)
    pooled = {}
    for label, group in groups.items():
        keys = wm.keys_for_manifest(group[0]["manifest"], group[0]["bidders"])
        pooled[label] = {"atlases": len(group)}
        for lam in PROBE_LAMBDAS:
            found = wm.detect_pooled([leaf_at(a, lam)["token_ids"] for a in group], keys)
            pooled[label][str(lam)] = {role: {k: found[role][k] for k in
                                              ("tokens_scored", "mean_score", "p_value")}
                                       for role in wm.ROLES}
    # Campaign audit: every distinct answer (leaf) of every atlas of a scenario, pooled,
    # with the same pooling on unwatermarked atlases of that scenario as the null.
    campaign = {}
    for label, group in groups.items():
        keys = wm.keys_for_manifest(group[0]["manifest"], group[0]["bidders"])
        texts = [leaf["token_ids"] for a in group for leaf in a["leaves"]]
        found = wm.detect_pooled(texts, keys)
        scenario = group[0]["manifest"].get("scenario")
        null_texts = [leaf["token_ids"] for _, a in null
                      if a["manifest"].get("scenario") == scenario for leaf in a["leaves"]]
        null_found = wm.detect_pooled(null_texts, keys) if null_texts else None
        campaign[label] = {
            "answers": len(texts),
            "watermarked": {r: {k: found[r][k] for k in ("tokens_scored", "mean_score", "p_value")}
                            for r in wm.ROLES},
            "unwatermarked_same_scenario": None if null_found is None else {
                r: {k: null_found[r][k] for k in ("tokens_scored", "mean_score", "p_value")}
                for r in wm.ROLES},
        }
    summary = {
        "alpha": alpha,
        "watermarked_atlases": len(rows),
        "null_atlases": len(nulls),
        "mean_detected_share": {r: float(np.mean([x["detected_share"][r] for x in rows]))
                                for r in wm.ROLES} if rows else None,
        "mean_drew_share": {r: float(np.mean([x["drew_share"][r] for x in rows]))
                            for r in wm.ROLES} if rows else None,
        "mean_false_alarm_share": {r: float(np.mean([x["false_alarm_share"][r] for x in nulls]))
                                   for r in wm.ROLES} if nulls else None,
        "robustness_detection_rate": {
            key: float(np.mean([p[key] for p in probes]))
            for key in ["clean"] + [f"{k}:{a}" for k, a in EDITS]
        } if probes else None,
    }
    summary["score_by_drawing_role"] = truth_table
    summary["pooled_by_lambda"] = pooled
    summary["campaign"] = campaign
    return {"summary": summary, "atlases": rows, "null": nulls, "probes": probes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--watermarked", nargs="+", default=[])
    parser.add_argument("--null", nargs="*", default=[])
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    marked = [(n, a) for n, a in atlases(args.watermarked) if "watermark" in a["manifest"]]
    null = [(n, a) for n, a in atlases(args.null) if "watermark" not in a["manifest"]]
    if not marked:
        print("no watermarked atlases found", file=sys.stderr)
        return 1
    report = evaluate(marked, null, args.alpha)
    print(json.dumps(report["summary"], indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
