"""Every ```bash block in README.md runs against the synthetic fixtures.

A block preceded by a line `<!-- skyhum-test: skip (reason) -->` is skipped (it needs data or network). Each
command must exit 0 unless its line ends with `# exit N`.
"""
import os
import re
import shlex
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = re.compile(r"<!--\s*skyhum-test:\s*skip")


def bash_blocks(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "```bash":
            prev = next((lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), "")
            j = i + 1
            while lines[j].strip() != "```":
                j += 1
            blocks.append(dict(line=i + 1, skip=bool(SKIP.search(prev)), body=lines[i + 1:j]))
            i = j
        i += 1
    return blocks


def commands(body):
    out, cur = [], ""
    for raw in body:
        line = raw.rstrip()
        if not cur and (not line.strip() or line.strip().startswith("#")):
            continue
        cur += line[:-1] + " " if line.endswith("\\") else line
        if not line.endswith("\\"):
            m = re.search(r"#\s*exit\s+([\d ,or]+)$", cur.strip())
            code = {int(c) for c in re.findall(r"\d+", m.group(1))} if m else {0}
            cmd = re.sub(r"\s+#.*$", "", cur).strip()
            out.append((cmd, code))
            cur = ""
    return out


BLOCKS = bash_blocks(os.path.join(ROOT, "README.md"))


def test_readme_has_runnable_blocks():
    assert sum(not b["skip"] for b in BLOCKS) >= 2


@pytest.mark.parametrize("block", BLOCKS, ids=[f"README.md:{b['line']}" for b in BLOCKS])
def test_readme_block(block, tmp_path_factory):
    if block["skip"]:
        pytest.skip("marked skyhum-test: skip (needs data or network)")
    cwd = tmp_path_factory.getbasetemp() / "readme_cwd"
    cwd.mkdir(exist_ok=True)
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""),
               SKYHUM_NO_GPU_PROBE="1", SKYHUM_WORKDIR=str(cwd / "workdir"))
    for cmd, want in commands(block["body"]):
        assert shlex.split(cmd)[0] == "skyhum", cmd
        p = subprocess.run(["bash", "-c", cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode in want, f"{cmd}\nexit {p.returncode}, wanted {want}\n{p.stdout[-800:]}\n{p.stderr[-800:]}"
