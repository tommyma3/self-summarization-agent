"""Recover explicit upstream assets and reserve leakage groups before collection.

This produces a proposal, never a loader-compatible executable task manifest.
No transcript parsing, downloads, task execution, or validation claims occur here.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

VERSION = 1
METADATA = ("origin", "instance_id", "repository_url", "base_commit", "issue_url")
FILES = ("instruction", "setup", "fail_to_pass", "pass_to_pass", "reference_fix", "verifier")
REQUIRED = METADATA + FILES
KINDS = {"upstream-release", "maintainer-supplied"}


def digest(path):
    with Path(path).open("rb") as stream:
        from hashlib import file_digest
        return file_digest(stream, "sha256").hexdigest()


def _evidence(field, record, root):
    if not isinstance(record, dict):
        raise ValueError("missing independently supplied evidence")
    if record.get("kind") not in KINDS:
        raise ValueError("evidence must be independent release/maintainer material")
    for key in ("source", "revision", "access", "path", "sha256"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"missing provenance {key}")
    path = Path(record["path"])
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("symlink evidence is unsupported")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("evidence file is missing or empty")
    if digest(path) != record["sha256"]:
        raise ValueError("evidence checksum mismatch")
    if field in METADATA:
        value = record.get("value")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing explicit metadata value")
        text = path.read_text().strip()
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            document = text
        observed = document.get(field) if isinstance(document, dict) else document
        if observed != value:
            raise ValueError("metadata value differs from independent evidence")
        if field == "base_commit" and not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("base commit must be a full Git SHA")
        if field in {"repository_url", "issue_url"} and not value.startswith("https://"):
            raise ValueError("expected canonical HTTPS source URL")
    return dict(record, path=str(path.resolve()))


def _keys(assets):
    """Conservative exact fingerprints; unresolved semantic overlap stays declared."""
    return sorted({
        "source:" + json.dumps([assets["origin"]["value"], assets["instance_id"]["value"]]),
        "issue:" + assets["issue_url"]["value"].rstrip("/"),
        "instruction:" + assets["instruction"]["sha256"],
        "patch:" + assets["reference_fix"]["sha256"],
    })


def recover(inventory_path):
    inventory_path = Path(inventory_path).resolve()
    body = json.loads(inventory_path.read_text())
    if (not isinstance(body, dict) or body.get("version") != VERSION or not body.get("dataset_revision")
            or not body.get("split_version") or type(body.get("split_seed")) is not int):
        raise ValueError("Expected versioned recovery inventory and split seed")
    candidates = body.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Expected candidate list")
    if any(not isinstance(c, dict) or not isinstance(c.get("assets", {}), dict) for c in candidates):
        raise ValueError("Expected candidate objects with asset mappings")
    identifiers = [c.get("candidate_id") for c in candidates]
    if any(not isinstance(i, str) or not i.strip() for i in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Candidate IDs must be nonempty and unique")
    recovered, excluded = [], []
    for candidate in candidates:
        assets, errors = {}, []
        for field in REQUIRED:
            try:
                assets[field] = _evidence(field, candidate.get("assets", {}).get(field), inventory_path.parent)
            except (ValueError, OSError) as exc:
                errors.append(dict(field=field, reason=str(exc)))
        if candidate.get("reserved_split") not in {"train", "eval"}:
            errors.append(dict(field="reserved_split", reason="reserve train/eval membership before collection"))
        if errors:
            excluded.append(dict(candidate_id=candidate["candidate_id"], exclusions=errors))
        else:
            recovered.append(dict(candidate_id=candidate["candidate_id"], assets=assets,
                reserved_split=candidate["reserved_split"], leakage_keys=_keys(assets)))

    # Join connected components, including transitive issue/content duplicates.
    groups = []
    for task in recovered:
        connected = [g for g in groups if set(task["leakage_keys"]) & g["keys"]]
        group = dict(keys=set(task["leakage_keys"]), tasks=[task])
        for previous in connected:
            group["keys"].update(previous["keys"])
            group["tasks"].extend(previous["tasks"])
            groups.remove(previous)
        groups.append(group)
    conflicts = []
    for group in groups:
        identity = sha256(json.dumps(sorted(group["keys"])).encode()).hexdigest()
        for task in group["tasks"]:
            task["leakage_group"] = identity
        if len({t["reserved_split"] for t in group["tasks"]}) > 1:
            conflicts.append(dict(leakage_group=identity, candidates=sorted(t["candidate_id"] for t in group["tasks"])))

    external = body.get("external_evaluation", [])
    overlaps, unresolved = [], []
    if not isinstance(external, list) or any(not isinstance(e, dict) for e in external):
        raise ValueError("Expected external evaluation objects")
    for evaluation in external:
        if not evaluation.get("name"):
            raise ValueError("External evaluation requires a name")
        index = evaluation.get("index")
        if index is None:
            unresolved.append(evaluation["name"])
            continue
        checked = _evidence("external_index", index, inventory_path.parent)
        entries = json.loads(Path(checked["path"]).read_text())
        if (not isinstance(entries, list) or not entries
                or any(not isinstance(e, dict) or not isinstance(e.get("leakage_keys"), list) or not e["leakage_keys"]
                       or not all(isinstance(k, str) and k.startswith(("source:", "issue:", "instruction:", "patch:"))
                                  for k in e["leakage_keys"]) for e in entries)):
            raise ValueError("External index needs nonempty explicit leakage_keys")
        keys = {k for entry in entries for k in entry["leakage_keys"]}
        for task in recovered:
            matches = sorted(keys & set(task["leakage_keys"]))
            if matches:
                overlaps.append(dict(candidate_id=task["candidate_id"], evaluation=evaluation["name"], keys=matches))
    blocked = {t for c in conflicts for t in c["candidates"]}
    blocked.update(o["candidate_id"] for o in overlaps)
    for group in groups:
        if any(t["candidate_id"] in blocked for t in group["tasks"]):
            blocked.update(t["candidate_id"] for t in group["tasks"])
    admitted = [t for t in recovered if t["candidate_id"] not in blocked]
    return dict(version=VERSION, dataset_revision=body["dataset_revision"],
        inventory_digest=digest(inventory_path), sources=body.get("sources", []),
        required_assets=list(REQUIRED), recovered_specifications=len(recovered),
        eligible_specifications=len(admitted), validated_upstream_tasks=0,
        gate="ready_for_bundle_validation" if admitted else "blocked_missing_independent_task_assets_or_split_conflict",
        candidates=sorted(recovered, key=lambda t: t["candidate_id"]), exclusions=excluded,
        split_proposal=dict(version=body["split_version"], seed=body["split_seed"],
            train=sorted(t["candidate_id"] for t in admitted if t["reserved_split"] == "train"),
            eval=sorted(t["candidate_id"] for t in admitted if t["reserved_split"] == "eval"),
            conflicts=conflicts, external_overlaps=overlaps,
            unresolved_external_evaluations=unresolved,
            limitation="Exact identity/content checks only; no semantic or full evaluation-disjointness claim."))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.resolve() == Path(args.inventory).resolve():
        parser.error("Output must not replace the recovery inventory")
    report = recover(args.inventory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    if not report["eligible_specifications"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
