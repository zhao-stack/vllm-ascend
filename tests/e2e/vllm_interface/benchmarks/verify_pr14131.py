# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent CPython witnesses for the three frozen PR14131 cases.

Only extracted parameter lists with literal defaults are compiled; production
function bodies and imports are never executed. This does not test GPU kernels.
"""

import argparse
import ast
import copy
import inspect
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--vllm-root", type=Path, required=True)
parser.add_argument("--ascend-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
UPSTREAM = args.vllm_root
ASCEND = args.ascend_root
OLD = "58d3918e3ea0a544ffedadad2ba84559e9c51d8f"
NEW = "cdc4824a21eaa986d4d1fee90a7e6465c9f706e6"
BASE = "b1038f13d729a96e183745610c52b997188b51e7"
ADAPTED = "175ef4d49c2072f9233619636f0309f4f88b4e2c"


def source(repo, sha, path):
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{sha}:{path}"], capture_output=True, check=True
    ).stdout.decode()


def main_lane_definition(text, name):
    tree = ast.parse(text)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != name:
            continue
        current = node
        allowed = True
        while current in parents:
            parent = parents[current]
            if isinstance(parent, ast.If) and isinstance(parent.test, ast.Call):
                test = parent.test
                if isinstance(test.func, ast.Name) and test.func.id == "vllm_version_is":
                    assert len(test.args) == 1 and ast.literal_eval(test.args[0]) == "0.27.1"
                    if current in parent.body:
                        allowed = False
            current = parent
        if allowed:
            matches.append(node)
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def signature(text, name):
    node = copy.deepcopy(main_lane_definition(text, name))
    for parameter in ast.walk(node.args):
        if isinstance(parameter, ast.arg):
            parameter.annotation = None
    for default in [*node.args.defaults, *(value for value in node.args.kw_defaults if value is not None)]:
        ast.literal_eval(default)
    node.decorator_list = []
    node.returns = None
    node.body = [ast.Pass()]
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<signature-witness>", "exec"),
        namespace,
    )
    return inspect.signature(namespace[name])


def bind(sig, arguments, keywords):
    try:
        sig.bind(*arguments, **keywords)
        return {"binds": True}
    except TypeError as error:
        return {"binds": False, "error": str(error)}


cases = {}
for name, up_file, down_file, keyword in (
    ("make_dummy", "vllm/v1/worker/gpu/input_batch.py", "vllm_ascend/worker/v2/input_batch.py", "max_query_len"),
    (
        "postprocess_mamba_fused_kernel",
        "vllm/v1/worker/mamba_utils.py",
        "vllm_ascend/ops/triton/mamba/postprocess.py",
        "TEMPORAL_TILES",
    ),
):
    signatures = {
        "old_upstream": signature(source(UPSTREAM, OLD, up_file), name),
        "new_upstream": signature(source(UPSTREAM, NEW, up_file), name),
        "baseline_downstream": signature(source(ASCEND, BASE, down_file), name),
        "adapted_downstream": signature(source(ASCEND, ADAPTED, down_file), name),
    }
    old = signatures["old_upstream"]
    required = [parameter for parameter in old.parameters.values() if parameter.default is inspect.Parameter.empty]
    arguments = [None] * len(required)
    assert keyword not in old.parameters
    assert keyword in signatures["new_upstream"].parameters
    before_old = bind(signatures["baseline_downstream"], arguments, {})
    before_new = bind(signatures["baseline_downstream"], arguments, {keyword: 1})
    new_accepts = bind(signatures["new_upstream"], arguments, {keyword: 1})
    after_new = bind(signatures["adapted_downstream"], arguments, {keyword: 1})
    assert before_old["binds"] and not before_new["binds"] and new_accepts["binds"] and after_new["binds"]
    cases[name] = {
        "evidence_kind": "interface_alignment",
        "signatures": {key: str(value) for key, value in signatures.items()},
        "baseline_old_call": before_old,
        "baseline_new_call": before_new,
        "new_upstream_call": new_accepts,
        "adapted_new_call": after_new,
    }

field_file = "vllm/model_executor/layers/fused_moe/router/base_router.py"
field_evidence = {}
for label, sha in (("old", OLD), ("new", NEW)):
    tree = ast.parse(source(UPSTREAM, sha, field_file))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseRouter")
    initializer = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignment = next(
        node for node in initializer.body if isinstance(node, ast.Assign) and ast.unparse(node) == "self.top_k = top_k"
    )
    field_evidence[label] = {"file": field_file, "line": assignment.lineno, "assignment": ast.unparse(assignment)}
cases["unchanged_top_k"] = {"evidence_kind": "negative_control", "endpoints": field_evidence}
payload = {"inputs": {"old": OLD, "new": NEW, "baseline": BASE, "adapted": ADAPTED}, "cases": cases, "passed": True}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
