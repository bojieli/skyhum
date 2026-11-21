"""Reports: JSON (machine) and a single self-contained HTML file (no external resources)."""
import html
import json
import os

SCHEMA_VERSION = 1


def _clean(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def document(command, result, exit_code):
    return _clean(dict(schema=SCHEMA_VERSION, command=command, exit_code=exit_code, result=result))


def dumps(doc):
    return json.dumps(doc, indent=2, sort_keys=False, ensure_ascii=False)


def write(path, doc):
    ext = os.path.splitext(path)[1].lower()
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    if ext == ".html":
        text = to_html(doc)
    elif ext == ".json":
        text = dumps(doc)
    else:
        from skyhum.errors import UsageError
        raise UsageError(f"--report must end in .json or .html (got {path})")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _rows(d, prefix=""):
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_rows(v, key + "."))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out.append((key, f"{len(v)} item(s), see below"))
        elif isinstance(v, list):
            out.append((key, "; ".join(str(x) for x in v)))
        else:
            out.append((key, "" if v is None else str(v)))
    return out


def to_html(doc):
    res = doc.get("result") or {}
    title = f"skyhum {doc.get('command', '')}"
    verdict = res.get("verdict") or res.get("result") or ""
    parts = [f"<h1>{html.escape(title)}</h1>"]
    if verdict:
        parts.append(f"<p class=v>{html.escape(str(verdict))}</p>")
    parts.append("<table>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
                                     for k, v in _rows(res)) + "</table>")
    for k, v in res.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            cols = list(v[0].keys())
            parts.append(f"<h2>{html.escape(k)}</h2><table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols)
                         + "</tr>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>"
                                                              for c in cols) + "</tr>" for r in v) + "</table>")
    css = ("body{font:14px/1.45 system-ui,sans-serif;margin:24px;max-width:960px;color:#1d1d1f;background:#fff}"
           "table{border-collapse:collapse;margin:8px 0 20px}th,td{border:1px solid #ccc;padding:3px 8px;"
           "text-align:left;vertical-align:top}th{background:#f3f3f3}.v{font-size:20px;font-weight:600}"
           "@media (prefers-color-scheme: dark){body{background:#161616;color:#e8e8e8}th{background:#262626}"
           "th,td{border-color:#444}}")
    return (f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{css}</style>"
            f"<body>{''.join(parts)}<p><small>Generated locally by skyhum {html.escape(str(res.get('version', '')))}. "
            f"No audio is embedded.</small></p></body></html>\n")
