"""One-time pre-migration step (already run on the old server; kept for provenance).

Adds an end-of-line marker comment to every line of EXECUTABLE code/config that depends on
the old server, and writes MIGRATION/server_paths.tsv (the full inventory).

Markers (all start with "[SERVER-"):
  [SERVER-PATH:REPO_ROOT]  /disk2/Yujin/adaptive_splitomc_tmc   -> new repository root
  [SERVER-PATH:DATA_ROOT]  /disk2/Yujin/datasets                -> new dataset root
  [SERVER-PATH:GIT_ROOT]   "/disk2/Yujin" (git dir for provenance) -> new git root
  [SERVER-PATH:EXTERNAL]   sibling projects adaptive_splitomc_v3 / _v4 (NOT migrated)
  [SERVER-GPU]             device / GPU-count assumptions of the old 2-GPU server

A marker is added only where a trailing comment cannot change behaviour:
  * .py : never on a line that starts or continues a multi-line string, never after a
          trailing backslash. Every modified file must keep an identical AST.
  * .sh : never inside a heredoc, never after a trailing backslash. `bash -n` must pass.
Lines that cannot be marked safely are still listed in the inventory (marked_inline = no).
Run records (runs/**.json, *.log, provenance/*.jsonl, queue files, reports) are NOT edited:
they are history and keep the old paths on purpose.
"""
import ast, io, re, subprocess, sys, tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = [  # (category, regex) — first match wins per occurrence
    ("SERVER-PATH:EXTERNAL", re.compile(r"/disk2/Yujin/adaptive_splitomc_v[0-9]+")),
    ("SERVER-PATH:REPO_ROOT", re.compile(r"/disk2/Yujin/adaptive_splitomc_tmc")),
    ("SERVER-PATH:DATA_ROOT", re.compile(r"/disk2/Yujin/datasets")),
    ("SERVER-PATH:GIT_ROOT", re.compile(r"[\"']/disk2/Yujin[\"']")),
    ("SERVER-PATH:HOME", re.compile(r"/home/ubuntu|/tmp/claude-\d+")),
]
GPU_RULES = [re.compile(p) for p in (r"CUDA_VISIBLE_DEVICES\s*=?\s*[\"']?0", r"SLOTS\s*=\s*\[.*cuda:", r"DEVICE1=.*cuda:1",
                                     r"def main\(device=\"cuda:1\"\)", r"else \"cuda:1\"\)",
                                     r"for i in [0-9 ]+; do launch .*r5_worker\.py")]
SKIP_DIRS = ("/runs/", "/provenance/", "/.git/", "/MIGRATION/")
CODE_EXT = {".py", ".sh", ".yaml", ".yml"}

def cats_for(line):
    out = []
    for cat, rx in RULES:
        if rx.search(line) and cat not in out:
            if cat == "SERVER-PATH:REPO_ROOT" and "adaptive_splitomc_v" in line and not re.search(r"adaptive_splitomc_tmc", line):
                continue
            out.append(cat)
    if any(rx.search(line) for rx in GPU_RULES):
        out.append("SERVER-GPU")
    return out

def py_unsafe_rows(src):
    """rows where a trailing comment would land inside a string (start/interior rows of multi-line strings)."""
    bad = set()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError):
        return None
    stack = []
    for t in toks:
        name = tokenize.tok_name[t.type]
        if name == "FSTRING_START":
            stack.append(t.start[0])
        elif name == "FSTRING_END" and stack:
            s = stack.pop()
            bad.update(range(s, t.end[0]))
        elif name == "STRING" and t.end[0] > t.start[0]:
            bad.update(range(t.start[0], t.end[0]))
    return bad

def sh_unsafe_rows(lines):
    bad, delim = set(), None
    for i, l in enumerate(lines, 1):
        if delim is not None:
            bad.add(i)
            if l.strip() == delim: delim = None
            continue
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", l)
        if m: delim = m.group(1)
    return bad

inventory = []   # file, line, category, marked_inline, text
changed = []
for f in sorted(ROOT.rglob("*")):
    if not f.is_file() or any(d in f.as_posix() for d in SKIP_DIRS):
        continue
    rel = f.relative_to(ROOT).as_posix()
    try:
        text = f.read_text()
    except (UnicodeDecodeError, OSError):
        continue
    if not re.search(r"/disk2/Yujin|/home/ubuntu|/tmp/claude-|cuda:|CUDA_VISIBLE", text):
        continue
    lines = text.split("\n")
    is_code = f.suffix in CODE_EXT
    unsafe = set()
    if f.suffix == ".py":
        unsafe = py_unsafe_rows(text)
        if unsafe is None:
            unsafe = set(range(1, len(lines) + 1))
    elif f.suffix == ".sh":
        unsafe = sh_unsafe_rows(lines)
    new = list(lines)
    for i, l in enumerate(lines, 1):
        cats = cats_for(l)
        cats += [t for t in re.findall(r"\[(SERVER-[A-Z:_]+)\]", l) if t not in cats]  # markers added by hand
        if not cats:
            continue
        cats = [c for c in cats if not (c == "SERVER-GPU" and not is_code)]
        if not cats:
            continue
        can = is_code and i not in unsafe and not l.rstrip().endswith("\\") and "[SERVER-" not in l
        if can:
            # a trailing "# ..." after code, after a string or after an existing comment is always
            # just a comment in .py/.sh/.yaml (multi-line strings, heredocs and "\" lines are excluded above)
            new[i - 1] = l.rstrip() + "  # " + " ".join(f"[{c}]" for c in cats)
        for c in cats:
            inventory.append((rel, i, c, "yes" if (can or "[SERVER-" in l) else "no", l.strip()[:160]))
    if new != lines:
        out = "\n".join(new)
        if f.suffix == ".py":
            if ast.dump(ast.parse(text)) != ast.dump(ast.parse(out)):
                sys.exit(f"AST changed in {rel}; aborting")
        f.write_text(out)
        if f.suffix == ".sh":
            r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
            if r.returncode:
                f.write_text(text); sys.exit(f"bash -n failed in {rel}: {r.stderr}")
        changed.append(rel)

inv = ROOT / "MIGRATION" / "server_paths.tsv"
with open(inv, "w") as fh:
    fh.write("file\tline\tcategory\tmarked_inline\ttext\n")
    for r in inventory:
        fh.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")
print(f"marked {len(changed)} files; inventory rows {len(inventory)} -> {inv.relative_to(ROOT)}")
for c in changed: print("  ", c)
