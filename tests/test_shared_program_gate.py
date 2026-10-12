"""Complete comparison rejects omissions and governance drift, not just samples."""
import ast
import json

import pytest

from scripts.verify_shared_program import digest, projection, verify_files


def contract(shared, permitted=None, sections=None):
    hashes = {path: digest(data) for path, data in shared.items()}
    return {"pair_verified": True, "shared_files": hashes,
            "shared_core_sha256": digest(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()),
            "permitted_installation_files": permitted or {},
            "permitted_installation_sections": sections or {}}


@pytest.mark.parametrize("path", [
    "aurora_lens/lens.py", "aurora_lens/pef/state.py", "aurora_lens/proxy/session.py",
    "aurora_lens/verify/checker.py", "governor/gate.py", "aurora_lens/data/policy.json",
])
def test_any_shared_program_change_fails(path):
    original = {path: b"original"}
    assert verify_files(contract(original), original, "private")["passed"]
    assert not verify_files(contract(original), {path: b"changed"}, "private")["passed"]


def test_missing_and_unexpected_files_are_not_ignored():
    original = {"aurora_lens/lens.py": b"core"}
    assert not verify_files(contract(original), {}, "private")["passed"]
    added = {**original, "aurora_lens/proxy/governance_hold.py": b"new code"}
    assert not verify_files(contract(original), added, "private")["passed"]


def test_installation_differences_require_exact_reviewed_variants():
    original = {"aurora_lens/lens.py": b"core"}
    variants = {"aurora_lens/cli.py": {
        "private": digest(b"old launcher"), "public": digest(b"managed launcher"),
        "reason": "Explicitly retained installation entrypoints."}}
    c = contract(original, variants)
    assert verify_files(c, {**original, "aurora_lens/cli.py": b"old launcher"}, "private")["passed"]
    assert verify_files(c, {**original, "aurora_lens/cli.py": b"managed launcher"}, "public")["passed"]
    assert not verify_files(c, {**original, "aurora_lens/cli.py": b"unexpected"}, "public")["passed"]


def test_installation_projection_does_not_exempt_the_rest_of_proxy():
    private = b"def hosted_edge():\n    return True\ndef govern():\n    return False\n"
    public = b"def hosted_edge():\n    return False\ndef govern():\n    return False\n"
    rule = {"function:hosted_edge": {
        "private": digest(ast.dump(ast.parse(private).body[0], include_attributes=False).encode()),
        "public": digest(ast.dump(ast.parse(public).body[0], include_attributes=False).encode()),
        "reason": "Only the explicitly approved installation helper may differ."}}
    core, errors = projection(private, rule, "private")
    assert not errors and projection(public, rule, "public")[0] == core
    c = contract({"aurora_lens/proxy/app.py": core}, sections={"aurora_lens/proxy/app.py": rule})
    assert verify_files(c, {"aurora_lens/proxy/app.py": public}, "public")["passed"]
    changed = public.replace(b"def govern():\n    return False", b"def govern():\n    return True")
    assert not verify_files(c, {"aurora_lens/proxy/app.py": changed}, "public")["passed"]
    changed_install = public.replace(b"def hosted_edge():\n    return False", b"def hosted_edge():\n    return None")
    assert not verify_files(c, {"aurora_lens/proxy/app.py": changed_install}, "public")["passed"]


def test_unverified_pair_is_not_a_release_contract():
    c = contract({"aurora_lens/lens.py": b"core"})
    c["pair_verified"] = False
    assert not verify_files(c, {"aurora_lens/lens.py": b"core"}, "private")["passed"]
