import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_reference_is_current():
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_cli_docs.py"), "--check"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_doc_links_exist():
    import re
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    for link in re.findall(r"\]\(((?:docs/|CONTRIBUTING|SECURITY)[^)#]*)\)", text):
        assert os.path.exists(os.path.join(ROOT, link)), link
