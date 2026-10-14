"""Publishing real generator output into the Atlas Explorer (synthetic_model_test).

The atlases here come from the REAL generator CLIs (token_auction_lab.atlas,
simplex_atlas, diffusion_atlas) run on tiny RANDOM models by
scripts/make_model_test_atlases.py. They check the pipeline generator -> JSON ->
scripts/publish_atlases.py -> explorer math, not model quality.
"""

import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
SAFE_ATLAS_PATH = re.compile(r"^[\w][\w.\-/]*\.json$")  # atlas.mjs safeAtlasPath

try:
    import torch  # noqa: F401
    from diffusers import UNet2DConditionModel  # noqa: F401
    from transformers import Qwen3Config  # noqa: F401

    HAVE_MODELS = True
except ImportError:
    HAVE_MODELS = False

STATE = {}


def setUpModule():
    if not HAVE_MODELS:
        return
    import torch

    from scripts import make_model_test_atlases as demo

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    tmp = tempfile.TemporaryDirectory()
    work = Path(tmp.name) / "runs"
    (work / "models").mkdir(parents=True)
    try:
        with redirect_stdout(StringIO()), patch("builtins.print"):
            runs = [
                demo.run_text(work / "text-run", scenario="transit-hotel", seeds=[3],
                              max_tokens=10, temperature=1.0, verify=1),
                demo.run_simplex(work / "simplex-run", scenario="three-hotels", seeds=[5],
                                 max_tokens=6, temperature=1.0, verify=2, max_cells=40),
                demo.run_image(work / "image-run", scene="cafe", seeds=[2], steps=4, size=16,
                               verify=2, work=work / "models"),
            ]
    finally:
        torch.set_num_threads(threads)
    dest = Path(tmp.name) / "atlas_data"
    dest.mkdir()
    shutil.copy(ROOT / "walkthrough/atlas_data/index.json", dest / "index.json")
    STATE.update(tmp=tmp, runs=runs, dest=dest)
    publish(runs, "--max-image-side", "8")


def tearDownModule():
    if "tmp" in STATE:
        STATE["tmp"].cleanup()


def publish(runs, *extra, dest=None):
    from scripts import publish_atlases

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return publish_atlases.main([*map(str, runs), "--dest", str(dest or STATE["dest"]),
                                     "--no-auto-tokenizer", *extra])


def published():
    index = json.loads((STATE["dest"] / "index.json").read_text())
    runs = {run.name for run in STATE["runs"]}
    entries = [e for e in index["atlases"] if e.get("run") in runs]
    return index, {e["path"]: json.loads((STATE["dest"] / e["path"]).read_text())
                   for e in entries}


def sources():
    """Source atlases by run name, as the generators wrote them."""
    out = {}
    for run in STATE["runs"]:
        for path in sorted(run.rglob("*.json")):
            data = json.loads(path.read_text())
            if data.get("schema") in ("bid-atlas-v1", "simplex-atlas-v1"):
                out[f"{run.name}/{path.relative_to(run).as_posix()}"] = data
    return out


@unittest.skipUnless(HAVE_MODELS, "Install the inference extra to run generator tests.")
class PublishTests(unittest.TestCase):
    def test_index_keeps_fixtures_and_adds_labeled_entries(self):
        index, atlases = published()
        self.assertEqual(index["schema"], "atlas-index-v1")
        original = json.loads((ROOT / "walkthrough/atlas_data/index.json").read_text())
        kept = original["atlases"]  # fixtures and the committed model-test runs
        self.assertTrue(any(e["path"].startswith("fixture-") for e in kept))
        self.assertEqual(index["atlases"][: len(kept)], kept)
        self.assertEqual(set(atlases), set(sources()))
        kinds = {(a["schema"], a["kind"]) for a in atlases.values()}
        self.assertEqual(kinds, {("bid-atlas-v1", "text"), ("bid-atlas-v1", "image"),
                                 ("simplex-atlas-v1", "text")})
        ids = [e["id"] for e in index["atlases"]]
        self.assertEqual(len(ids), len(set(ids)))
        for entry in index["atlases"][len(kept):]:
            atlas = atlases[entry["path"]]
            self.assertRegex(entry["path"], SAFE_ATLAS_PATH)
            self.assertEqual(entry["mode"], "synthetic_model_test")
            self.assertEqual(atlas["mode"], "synthetic_model_test")
            for key in ("schema", "kind", "bidders", "seed"):
                self.assertEqual(entry[key], atlas[key])
            self.assertEqual(entry["leaves"], len(atlas["leaves"]))
            self.assertTrue(entry["title"].startswith(atlas["title"]))
            self.assertIn(f"seed {atlas['seed']}", entry["title"])

    def test_content_is_copied_unchanged_except_recorded_display_fields(self):
        _, atlases = published()
        for path, source in sources().items():
            atlas = atlases[path]
            notes = atlas.pop("published")["changes"]
            if source["schema"] == "simplex-atlas-v1":
                source["leaves"] = source.pop("cells")
                self.assertIn("cells renamed to leaves (MOONSHOTS.md section 4)", notes)
            else:
                self.assertEqual(notes, [])
            self.assertEqual(atlas, source, path)
            text = json.dumps(atlas)
            self.assertNotIn(str(STATE["dest"].parent), text)  # no temporary paths

    def test_explorer_display_fields_come_from_the_generators(self):
        from token_auction_lab.prompts import scenarios
        from token_auction_lab.prompts_n import scenarios_n

        _, atlases = published()
        by_kind = {(a["schema"], a["kind"]): a for a in atlases.values()}
        text = by_kind[("bid-atlas-v1", "text")]
        self.assertEqual(text["title"], scenarios()["transit-hotel"]["title"])
        self.assertEqual(text["manifest"]["user_prompt"], scenarios()["transit-hotel"]["task"])
        leaves = {leaf["id"]: leaf for leaf in text["leaves"]}
        self.assertTrue(text["splits"])
        for split in text["splits"]:
            for side, leaf in (("b", leaves[split["parent"]]), ("a", leaves[split["child"]])):
                piece = split[f"{side}_token_text"]
                self.assertIsInstance(piece, str)
                self.assertTrue(piece)
                if piece != "<|im_end|>":
                    self.assertTrue(leaf["text"].startswith(split["prefix_text"] + piece))
        simplex = by_kind[("simplex-atlas-v1", "text")]
        self.assertEqual(simplex["title"], scenarios_n()["three-hotels"]["title"])
        self.assertEqual(simplex["manifest"]["user_prompt"], scenarios_n()["three-hotels"]["task"])
        cells = {leaf["id"]: leaf for leaf in simplex["leaves"]}
        for split in simplex["splits"]:
            self.assertAlmostEqual(sum(split["apex"]), 1.0, places=12)
            self.assertEqual([p["cell"] for p in split["pieces"]], split["children"])
            for piece in split["pieces"]:
                cell = cells[piece["cell"]]
                self.assertEqual(cell["steps"][split["step"]]["winner"], piece["winner"])
                self.assertEqual(cell["token_ids"][split["step"]], piece["token"])
                if piece["token_text"] != "<|im_end|>":
                    self.assertTrue(cell["text"].startswith(split["prefix_text"]
                                                            + piece["token_text"]))
        image = by_kind[("bid-atlas-v1", "image")]
        self.assertEqual(image["title"], "Street cafe terrace")
        self.assertTrue(image["manifest"]["prompts"]["shared"])

    def test_images_are_relative_to_the_json_and_downscaled(self):
        from PIL import Image

        _, atlases = published()
        images = 0
        for path, atlas in atlases.items():
            if atlas["kind"] != "image":
                continue
            resized = atlas["published"]["resized_images"]
            for leaf in atlas["leaves"]:
                rel = leaf["image"]
                self.assertFalse(rel.startswith("/") or ".." in rel)
                file = STATE["dest"] / Path(path).parent / rel
                with Image.open(file) as img:
                    self.assertLessEqual(max(img.size), 8)
                self.assertEqual(resized[rel]["original_size"], [16, 16])
                images += 1
        self.assertGreater(images, 1)

    def test_refusals_and_republish(self):
        from scripts.publish_atlases import PublishError

        image_run = next(r for r in STATE["runs"] if r.name == "image-run")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "dest"
            dest.mkdir()
            source = next(image_run.glob("*.atlas.json"))

            def variant(name, change):
                run = root / name
                shutil.copytree(image_run, run)
                data = json.loads(source.read_text())
                change(data)
                (run / source.name).write_text(json.dumps(data))
                return run

            relabeled = variant("unknown-mode", lambda d: d.update(mode="live"))
            with self.assertRaisesRegex(PublishError, "Unknown mode"):
                publish([relabeled], dest=dest)
            escape = variant("escape", lambda d: d["leaves"][0].update(image="../../x.jpg"))
            with self.assertRaisesRegex(PublishError, "Unsafe image"):
                publish([escape], dest=dest)

            def truncate(d):
                d["complete"] = False
                d["leaves"] = d["leaves"][:-1]

            partial = variant("partial", truncate)
            publish([partial], dest=dest)  # skipped with a warning, nothing written
            self.assertFalse((dest / "partial").exists())
            publish([partial], "--allow-incomplete", dest=dest)
            self.assertTrue((dest / "partial").is_dir())
            (dest / "handmade").mkdir()
            with self.assertRaisesRegex(PublishError, "refusing to overwrite"):
                publish([image_run], "--name", "handmade", dest=dest)
            publish([image_run], dest=dest)
            publish([image_run], dest=dest)  # republish replaces, never duplicates
            index = json.loads((dest / "index.json").read_text())
            paths = [e["path"] for e in index["atlases"]]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertEqual(sum(p.startswith("image-run/") for p in paths), 1)
            self.assertTrue(any(p.startswith("partial/") for p in paths))


@unittest.skipUnless(HAVE_MODELS and shutil.which("node"),
                     "Needs the inference extra and Node for the browser-math check")
class ExplorerMathOnGeneratorOutputTests(unittest.TestCase):
    def test_atlas_mjs_matches_generator_payments_and_lookup(self):
        from token_auction_lab.atlas import leaf_for, payments
        from token_auction_lab.diffusion_atlas import leaf_charges
        from token_auction_lab.simplex_atlas import cell_payments, random_interior_point

        _, atlases = published()
        rng = np.random.default_rng(11)
        pair, lookups, race = [], [], []
        for path, atlas in atlases.items():
            leaves = atlas["leaves"]
            if atlas["schema"] == "simplex-atlas-v1":
                for i, leaf in enumerate(leaves):
                    poly = np.array(leaf["polygon"])
                    center = poly.mean(axis=0)
                    points = [center] + [random_interior_point(poly, rng) for _ in range(3)]
                    for k, point in enumerate(points):
                        total = float(rng.choice([1.0, 10.0, 37.5]))
                        bids = [total * float(x) for x in point]
                        race.append({"atlas": path, "leaf": i, "point": point.tolist(),
                                     "S": total, "inside": k == 0,
                                     "charges": cell_payments(leaf["steps"], bids)})
                continue
            for i, leaf in enumerate(leaves):
                lams = [0.5 * (leaf["lo"] + leaf["hi"]), leaf["hi"],
                        math.nextafter(leaf["lo"], 1.0), *rng.uniform(leaf["lo"], leaf["hi"], 3)]
                for lam in map(float, lams):
                    if not leaf["lo"] < lam <= leaf["hi"]:
                        continue
                    total = float(rng.choice([1.0, 2.0, 100.0, 37.5]))
                    bid_a, bid_b = total * lam, total * (1.0 - lam)
                    if bid_a / (bid_a + bid_b) != lam:
                        continue  # bids would round to a ratio across a leaf boundary
                    if atlas["kind"] == "text":
                        ref = payments(leaf, bid_a, bid_b)
                        a, b = ref["payment_a"], ref["payment_b"]
                    else:
                        a, b = leaf_charges(leaf["steps"], bid_a, bid_b)
                    pair.append({"atlas": path, "leaf": i, "lambda": lam, "S": total,
                                 "a": a, "b": b})
                    if atlas["kind"] == "text":
                        lookups.append({"atlas": path, "lambda": lam,
                                        "id": leaf_for(atlas, lam)["id"]})
                if atlas["kind"] == "text" and leaf["lo"] > 0:
                    lookups.append({"atlas": path, "lambda": leaf["lo"],
                                    "id": leaf_for(atlas, leaf["lo"])["id"]})
        self.assertGreater(len(pair), 20)
        self.assertGreater(len(race), 20)
        self.assertTrue(any(c["a"] > 0 for c in pair) and any(c["b"] > 0 for c in pair))
        script = """
import fs from 'fs';
import assert from 'assert';
import {chargesAt, leafIndexAt, raceCharges, cellIndexAt, partitionProblems, promptText,
        atlasLeaves, tickRows, imageAspect, outcomeKeys} from '%s';
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const cache = {};
const get = p => (cache[p] ||= JSON.parse(fs.readFileSync(data.dest + '/' + p, 'utf8')));
const close = (x, y) => Math.abs(x - y) <= 1e-12 * Math.max(1, Math.abs(y));
for (const c of data.pair) {
  const r = chargesAt(get(c.atlas).leaves[c.leaf], c.lambda, c.S);
  assert.ok(close(r.a, c.a), `A ${c.atlas} ${c.leaf} ${c.lambda}: ${r.a} vs ${c.a}`);
  assert.ok(close(r.b, c.b), `B ${c.atlas} ${c.leaf} ${c.lambda}: ${r.b} vs ${c.b}`);
}
for (const l of data.lookups) {
  const leaves = get(l.atlas).leaves;
  assert.strictEqual(leaves[leafIndexAt(leaves, l.lambda)].id, l.id, JSON.stringify(l));
}
for (const r of data.race) {
  const leaves = atlasLeaves(get(r.atlas));
  if (r.inside) assert.strictEqual(cellIndexAt(leaves, r.point), r.leaf);
  const got = raceCharges(leaves[r.leaf], r.point, r.S).charges;
  got.forEach((v, i) => assert.ok(close(v, r.charges[i]),
                                  `race ${r.leaf} ${i}: ${v} vs ${r.charges[i]}`));
}
for (const p of data.paths) {
  const atlas = get(p);
  if (atlas.schema === 'bid-atlas-v1') assert.deepStrictEqual(partitionProblems(atlas.leaves), []);
  if (atlas.kind === 'text') {
    assert.strictEqual(promptText(atlas.manifest), atlas.manifest.user_prompt);
  }
  else assert.ok(promptText(atlas.manifest).length > 10);
  for (const k of outcomeKeys(atlas.leaves)) assert.ok(!k.endsWith('argmax'));
}
assert.deepStrictEqual(atlasLeaves({cells: [1]}), [1]);
assert.strictEqual(promptText({prompts: [{text: 'x'}], user_prompt: 'Hi'}), 'Hi');
assert.strictEqual(promptText({prompts: {shared: 'a scene'}}), 'a scene');
assert.deepStrictEqual(tickRows([0, 5, 10, 30, 31, 32, 33], 20, 3), [0, 1, 2, 0, 1, 2, -1]);
assert.strictEqual(imageAspect({width: 1024, height: 1024}), 1);
assert.strictEqual(imageAspect({image_size: [640, 400]}), 1.6);
assert.strictEqual(imageAspect({}), 1.6);
const clip = {'clip:A': 0.2, clip_argmax: 'A', 'clip_top:A': true};
assert.deepStrictEqual(outcomeKeys([{outcomes: clip}]), ['clip:A', 'clip_top:A']);
""" % (ROOT / "walkthrough" / "atlas.mjs").as_uri()
        payload = {"dest": str(STATE["dest"]), "pair": pair, "lookups": lookups, "race": race,
                   "paths": list(atlases)}
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script],
                                input=json.dumps(payload), text=True, cwd=ROOT,
                                capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


CHECKER = r'''
"""Fake `python` for run_moonshots.sh: parse each command with the real CLI, run nothing."""
import json, os, subprocess, sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.environ["REPO_ROOT"])
args = sys.argv[1:]
if args[:1] == ["-c"]:
    raise SystemExit(subprocess.call([sys.executable, *args]))


class Parsed(Exception):
    pass


record = {"argv": args}
if args[:1] == ["-m"]:
    module, argv = args[1], args[2:]
    record["module"] = module
    if module == "token_auction_lab.atlas":
        from token_auction_lab import atlas

        def stop(parsed):
            raise Parsed(vars(parsed))

        try:
            with patch("token_auction_lab.atlas._load", side_effect=stop):
                atlas.main(argv)
        except Parsed as done:
            record["parsed"] = done.args[0]
    elif module == "token_auction_lab.simplex_atlas":
        from token_auction_lab import simplex_atlas

        def stop(parsed, pinned, seed):
            raise Parsed(vars(parsed))

        try:
            with patch("token_auction_lab.cli.load_model", side_effect=stop):
                simplex_atlas.main(argv)
        except Parsed as done:
            record["parsed"] = done.args[0]
    elif module == "token_auction_lab.diffusion_atlas":
        from token_auction_lab import diffusion_atlas

        record["parsed"] = vars(diffusion_atlas.build_parser().parse_args(argv))
    out = record.get("parsed", {}).get("output")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)  # so the publish stage has a run dir
elif args[:1] == ["scripts/publish_atlases.py"]:
    from scripts import publish_atlases

    record["module"] = "publish"
    record["returncode"] = publish_atlases.main(args[1:] + ["--dest", os.environ["FAKE_DEST"]])
with open(os.environ["SHIM_LOG"], "a") as log:
    log.write(json.dumps(record, default=str) + "\n")
'''


@unittest.skipUnless(shutil.which("bash"), "bash is needed to dry-run the runbook")
class RunbookDryRunTests(unittest.TestCase):
    def test_every_stage_uses_flags_the_real_clis_accept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "checker.py").write_text(CHECKER)
            shim = root / "bin" / "python"
            shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{root / "checker.py"}" "$@"\n')
            shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
            (root / "dest").mkdir()
            env = {**os.environ, "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
                   "OUT": str(root / "out"), "SHIM_LOG": str(root / "log.jsonl"),
                   "FAKE_DEST": str(root / "dest"), "REPO_ROOT": str(ROOT),
                   "TEXT_SEEDS": "1 2", "SIMPLEX_SEEDS": "1 2", "IMAGE_SEEDS": "1"}
            result = subprocess.run(["bash", "scripts/run_moonshots.sh"], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = [json.loads(line) for line in (root / "log.jsonl").read_text().splitlines()]
        by_module = {}
        for record in records:
            by_module.setdefault(record.get("module"), []).append(record)
        text = by_module["token_auction_lab.atlas"]
        self.assertEqual(len(text), 3)
        self.assertTrue(all("parsed" in r for r in text), text)
        self.assertEqual([r["parsed"]["live_defaults"] for r in text], [True, False, False])
        self.assertEqual([r["parsed"]["scenario"] for r in text[1:]],
                         ["transit-hotel", "competing-hotels"])
        for r in text:
            self.assertTrue(r["parsed"]["local_files_only"])
            self.assertEqual(r["parsed"]["seeds"], [1, 2])
        (simplex,) = by_module["token_auction_lab.simplex_atlas"]
        self.assertEqual(simplex["parsed"]["scenario"], "three-hotels")
        self.assertEqual(simplex["parsed"]["time_budget_seconds"], 600.0)  # 1200 s / 2 seeds
        (image,) = by_module["token_auction_lab.diffusion_atlas"]
        self.assertEqual(image["parsed"]["model"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(image["parsed"]["scene"], "cafe")
        self.assertEqual(image["parsed"]["steps"], 30)
        publishes = by_module["publish"]
        self.assertEqual(len(publishes), 3)  # text, simplex, image; no atlases yet, all skipped
        self.assertTrue(all(r["returncode"] == 0 for r in publishes))
        self.assertTrue(all("--skip-missing" in r["argv"] for r in publishes))


if __name__ == "__main__":
    unittest.main()
