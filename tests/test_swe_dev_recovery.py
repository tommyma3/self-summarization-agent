"""Recovery is an admission proposal, not fresh policy or Docker evidence."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from self_summarization_agent.benchmarks.swe_dev.recovery import FILES, METADATA, recover


def inventory(tmp_path):
    assets = {}
    values = dict(origin="swe-dev", instance_id="upstream-1", repository_url="https://example.org/repo",
                  base_commit="a" * 40, issue_url="https://example.org/repo/issues/1")
    for field in METADATA + FILES:
        payload = (values.get(field, field) + "\n").encode()
        path = tmp_path / field
        path.write_bytes(payload)
        assets[field] = dict(path=field, sha256=sha256(payload).hexdigest(),
            kind="maintainer-supplied", source="https://example.org/recovered", revision="release-1",
            access="local supplied assets")
        if field in METADATA:
            assets[field]["value"] = values[field]
    return dict(version=1, dataset_revision="test-release", split_version="pilot-v1", split_seed=3,
                candidates=[dict(candidate_id="upstream:1", assets=assets, reserved_split="train")],
                external_evaluation=[dict(name="SWE-bench Verified", index=None)])


def run(tmp_path, body):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(body))
    return recover(path)


def test_no_assets_is_explicit_block_and_never_an_executable_manifest(tmp_path):
    body = inventory(tmp_path)
    body["candidates"] = []
    report = run(tmp_path, body)
    assert report["eligible_specifications"] == report["validated_upstream_tasks"] == 0
    assert report["gate"].startswith("blocked")
    assert report["split_proposal"]["train"] == report["split_proposal"]["eval"] == []
    assert "tasks" not in report


def test_complete_evidence_reserves_split_without_claiming_validation(tmp_path):
    report = run(tmp_path, inventory(tmp_path))
    assert report["eligible_specifications"] == 1
    assert report["validated_upstream_tasks"] == 0
    assert report["gate"] == "ready_for_bundle_validation"
    assert report["split_proposal"]["train"] == ["upstream:1"]
    assert report["split_proposal"]["unresolved_external_evaluations"] == ["SWE-bench Verified"]
    assert report == run(tmp_path, json.loads((tmp_path / "inventory.json").read_text()))


@pytest.mark.parametrize("mutation, field", [
    ("missing", "setup"), ("checksum", "reference_fix"), ("transcript", "instruction"),
    ("blank", "fail_to_pass"), ("commit", "base_commit"), ("symlink", "verifier"),
    ("split", "reserved_split"), ("revision", "pass_to_pass"),
])
def test_excludes_incomplete_or_untrusted_evidence(tmp_path, mutation, field):
    body = inventory(tmp_path)
    candidate = body["candidates"][0]
    if mutation == "missing": del candidate["assets"][field]
    elif mutation == "checksum": (tmp_path / field).write_text("changed")
    elif mutation == "transcript": candidate["assets"][field]["kind"] = "assistant-transcript"
    elif mutation == "blank": (tmp_path / field).write_text("")
    elif mutation == "commit": candidate["assets"][field]["value"] = "a123"
    elif mutation == "symlink":
        (tmp_path / "linked").symlink_to(tmp_path / field)
        candidate["assets"][field]["path"] = "linked"
    elif mutation == "split": candidate.pop("reserved_split")
    elif mutation == "revision": candidate["assets"][field].pop("revision")
    report = run(tmp_path, body)
    assert report["eligible_specifications"] == 0
    assert report["exclusions"][0]["candidate_id"] == "upstream:1"
    assert field in {e["field"] for e in report["exclusions"][0]["exclusions"]}


def test_transitive_content_overlap_cannot_cross_reserved_splits(tmp_path):
    body = inventory(tmp_path)
    first = body["candidates"][0]
    second = deepcopy(first)
    second.update(candidate_id="upstream:2", reserved_split="eval")
    second["assets"]["instance_id"]["value"] = "upstream-2"
    second["assets"]["issue_url"]["value"] += "2"
    body["candidates"].append(second)
    for field in ("instance_id", "issue_url"):
        record = second["assets"][field]
        path = tmp_path / (field + "-second")
        path.write_text(record["value"])
        record.update(path=path.name, sha256=sha256(path.read_bytes()).hexdigest())
    report = run(tmp_path, body)
    assert report["recovered_specifications"] == 2
    assert report["eligible_specifications"] == 0
    assert report["split_proposal"]["conflicts"][0]["candidates"] == ["upstream:1", "upstream:2"]


def test_external_evaluation_overlap_blocks_whole_leakage_group(tmp_path):
    body = inventory(tmp_path)
    second = deepcopy(body["candidates"][0])
    second["candidate_id"] = "upstream:2"
    second["assets"]["issue_url"]["value"] += "2"
    second["assets"]["instance_id"]["value"] += "2"
    for field in ("instance_id", "issue_url"):
        record = second["assets"][field]
        path = tmp_path / (field + "-second")
        path.write_text(record["value"])
        record.update(path=path.name, sha256=sha256(path.read_bytes()).hexdigest())
    body["candidates"].append(second)
    index = tmp_path / "external.json"
    index.write_text(json.dumps([dict(leakage_keys=["issue:https://example.org/repo/issues/1"])]))
    body["external_evaluation"][0]["index"] = dict(path="external.json", sha256=sha256(index.read_bytes()).hexdigest(),
        kind="upstream-release", source="https://example.org/eval", revision="eval-v1", access="public")
    report = run(tmp_path, body)
    assert report["eligible_specifications"] == 0
    assert report["split_proposal"]["external_overlaps"][0]["evaluation"] == "SWE-bench Verified"


def test_duplicate_candidate_ids_fail_closed(tmp_path):
    body = inventory(tmp_path)
    body["candidates"] *= 2
    with pytest.raises(ValueError, match="unique"):
        run(tmp_path, body)


def test_metadata_cannot_be_guessed_from_another_artifact(tmp_path):
    body = inventory(tmp_path)
    body["candidates"][0]["assets"]["instance_id"]["value"] = "invented-id"
    report = run(tmp_path, body)
    assert report["eligible_specifications"] == 0
    assert "differs from independent evidence" in report["exclusions"][0]["exclusions"][0]["reason"]


def test_connected_groups_merge_through_different_keys(tmp_path):
    body = inventory(tmp_path)
    first = body["candidates"][0]
    second, bridge = deepcopy(first), deepcopy(first)
    second.update(candidate_id="upstream:2", reserved_split="eval")
    bridge.update(candidate_id="upstream:3")
    for task, suffix in ((second, "second"), (bridge, "bridge")):
        for field in ("instance_id", "issue_url", "instruction", "reference_fix"):
            record = task["assets"][field]
            content = suffix + field
            if field in METADATA:
                content = record["value"] + suffix
                record["value"] = content
            path = tmp_path / (suffix + field)
            path.write_text(content)
            record.update(path=path.name, sha256=sha256(path.read_bytes()).hexdigest())
    # first -> bridge via issue; bridge -> second via patch. first and second
    # share no key, so a pairwise split comparison would miss this overlap.
    bridge["assets"]["issue_url"] = deepcopy(first["assets"]["issue_url"])
    bridge["assets"]["reference_fix"] = deepcopy(second["assets"]["reference_fix"])
    body["candidates"] += [second, bridge]
    report = run(tmp_path, body)
    assert report["recovered_specifications"] == 3
    assert report["eligible_specifications"] == 0
    assert len(report["split_proposal"]["conflicts"][0]["candidates"]) == 3


def test_cli_publishes_blocked_report_and_preserves_inventory(tmp_path, monkeypatch):
    import sys
    from self_summarization_agent.benchmarks.swe_dev.recovery import main
    body = inventory(tmp_path)
    body["candidates"] = []
    run(tmp_path, body)
    source, output = tmp_path / "inventory.json", tmp_path / "report.json"
    original = source.read_bytes()
    monkeypatch.setattr(sys, "argv", ["recovery", str(source), "--output", str(output)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
    assert json.loads(output.read_text())["eligible_specifications"] == 0
    assert source.read_bytes() == original
    monkeypatch.setattr(sys, "argv", ["recovery", str(source), "--output", str(source)])
    with pytest.raises(SystemExit):
        main()
    assert source.read_bytes() == original
