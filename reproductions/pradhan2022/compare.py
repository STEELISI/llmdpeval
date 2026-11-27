"""Compare each executed notebook in outputs/ with the outputs the authors saved in original/.
Per code cell: text outputs are normalised (timings, tqdm bars, object addresses removed, predicate lists
sorted because an explanation is a set), then compared
token by token; numbers must agree within --rtol. Prints MATCH/DIFF per cell and an overall verdict.
Usage: python compare.py [--rtol 1e-6]"""
import argparse, json, re, sys
from pathlib import Path

NOISE = [r"CPU times:.*", r"Wall time:.*", r".*it/s\]", r".*s/it\]", r"at 0x[0-9a-f]+", r"\r"]
NUM = re.compile(r"-?\d+\.?\d*(?:e-?\d+)?")

def cell_texts(nb):
    out = []
    for c in json.load(open(nb, encoding="utf-8"))["cells"]:
        if c["cell_type"] != "code": continue
        t = "".join("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
                    for o in c.get("outputs", []))
        for pat in NOISE: t = re.sub(pat, "", t)
        # An explanation is a SET of predicates: sort each ["a=1", "b=0"] list so order does not count.
        t = re.sub(r'\[("[^"\]]*"(?:, "[^"\]]*")*)\]', lambda m: "[" + ", ".join(sorted(m.group(1).split(", "))) + "]", t)
        out.append((c.get("execution_count"), "".join(c["source"])[:60].replace("\n", " "), t.strip()))
    return out

def same(a, b, rtol):
    if NUM.sub("#", a).split() != NUM.sub("#", b).split(): return False, "text differs"
    worst = 0.0
    for x, y in zip(map(float, NUM.findall(a)), map(float, NUM.findall(b))):
        worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-12))
    return worst <= rtol, f"max rel diff {worst:.2e}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rtol", type=float, default=1e-6); a = ap.parse_args()
    ok_all = True
    for ex in sorted(Path("outputs").glob("*.executed.ipynb")):
        orig = Path("original/explanations") / ex.name.replace(".executed", "")
        print(f"== {orig.name}")
        for (n, src, want), (_, _, got) in zip(cell_texts(orig), cell_texts(ex)):
            if not want: continue            # the authors saved no output for this cell
            ok, why = same(want, got, a.rtol); ok_all &= ok
            print(f"  [{n}] {'MATCH' if ok else 'DIFF '} {why:22} {src}")
            if not ok: print("     authors: " + want[:300].replace("\n", " | ") + "\n     ours:    " + got[:300].replace("\n", " | "))
    print("VERDICT:", "MATCH" if ok_all else "DIFF (see above)")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
