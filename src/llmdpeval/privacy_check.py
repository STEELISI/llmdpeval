"""Static privacy check of a generated script: high-precision rules drawn from failures seen in hand-graded
scripts. Used as repair feedback for scripts that run, and by app_grade as flags. A flag is a reason to look, not a
grade: each rule was measured on hand-graded scripts and kept only if it had no false positive there.

  check(code, stdout=None, budget=1.0, delta_zero=True, replace_one=True) -> [{"rule", "line", "detail", "message"}]
  feedback(flags) -> one text block the LLM can act on

The analysis is a small flow-sensitive taint pass over the AST, per function, in statement order. Labels:
  npw   weights of a model trained without DP (sklearn estimator, torch/numpy training loop with no noise)
  npd   anything computed from npw (predictions, accuracies)
  scaled  statistics of the private data (min/max/std/quantile, fitted scalers) and what is computed from them
  noisy   output of a DP mechanism or a noise draw
  mask / subset / gsize   a boolean row filter, rows selected by it, and the size of such a selection
"""
import ast
import json
import re

NOISE_ATTRS = {"laplace", "normal", "gamma", "exponential", "gumbel", "geometric", "randomise", "randomize",
               "standard_normal", "normal_", "release", "invoke"}
SKLEARN_MODEL_MODULES = ("sklearn.linear_model.", "sklearn.svm.", "sklearn.ensemble.", "sklearn.tree.",
                         "sklearn.neighbors.", "sklearn.naive_bayes.", "sklearn.neural_network.")
SCALERS = {"StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "scale", "minmax_scale", "robust_scale",
           "maxabs_scale", "QuantileTransformer", "PowerTransformer"}
STAT_ATTRS = {"min", "max", "std", "quantile", "nanmin", "nanmax", "nanstd", "ptp", "percentile", "nanpercentile",
              "nanquantile", "amin", "amax", "var", "nanvar"}
DPL_MEAN_LIKE = {"mean", "nanmean", "var", "nanvar", "std", "nanstd"}  # diffprivlib: n of the input is public
DPL_NEED_BOUNDS = DPL_MEAN_LIKE | {"sum", "nansum", "median", "quantile", "percentile"}
MESSAGES = {
    "nonprivate_model": "A model is trained on private rows without DP ({detail}). The trained model is itself a "
                        "release, and accuracies or other values computed from its predictions are not private even "
                        "if noise is added afterwards. Train the released model with a DP mechanism (e.g. diffprivlib "
                        "LogisticRegression with data_norm and public bounds, or objective/output perturbation) and "
                        "compute every DP output from that model.",
    "private_bounds": "Bounds, scaling or standardisation computed from the private data ({detail}) feed a DP "
                      "mechanism, so its sensitivity depends on individual rows. Use only the public bounds/domains "
                      "given in the task to clip and scale.",
    "missing_bounds": "{detail} is called without public bounds, so diffprivlib takes them from the data (a privacy "
                      "leak). Pass the public bounds (data_norm for LogisticRegression, bounds_X/bounds_y for "
                      "LinearRegression, bounds for tools).",
    "group_size_public": "The size of a private subgroup ({detail}) is used as if it were public. Only the total number "
                         "of rows is public: release a noisy count for each group (or a noisy numerator and a noisy "
                         "denominator) and divide the noisy values.",
    "add_remove_calibration": "{detail}: this is calibrated for add/remove-one neighbours, but the task uses replace-one "
                              "(the number of rows is public). Under replace-one a histogram or count changes by up to "
                              "2 in L1 (use d_in=2 / contributions=2), and per-group releases do not compose in "
                              "parallel (a row can move between groups): add their epsilons.",
    "unused_noise": "Noise is drawn into `{detail}` but never used, so the value released in its place has no noise. "
                    "Add the noise to the released value.",
    "positive_delta": "The task requires delta = 0, but the code sets {detail}. Use a pure-DP mechanism (Laplace, "
                      "discrete Laplace, exponential) with delta = 0.",
    "dp_equals_true": "The output's DP value equals the true value for {detail}: that release has no noise.",
    "over_budget": "The output's epsilon_total is {detail}, over the budget of epsilon = {budget:g}.",
}


def _qual(node, aliases):
    """Dotted name of a Name/Attribute chain with import aliases resolved, else None."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qual(node.value, aliases)
        return f"{base}.{node.attr}" if base else None
    return None


def _aliases(tree):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                out[a.asname or a.name] = f"{n.module}.{a.name}"
    return out


def _attr(call):
    return call.func.attr if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) else None


def _const_num(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


WEIGHT_NAME = re.compile(r"(?i)^(w|ws|weights?|theta|beta|coefs?|params?|model|clf|w_\w+|\w+_w|\w*model\w*)$")
TRAIN_TEXT = re.compile(r"\.backward\(\)|torch\.optim|\bminimize\(|\blstsq\(|linalg\.solve\(|^\s*(w|weights?|theta|beta|coef)\w*\s*-=|\.fit\(", re.M)
SCALE_KWARGS = {"bounds", "bounds_X", "bounds_y", "bounds_x", "data_norm", "lower", "upper", "scale", "sensitivity",
                "clip", "clip_norm", "norm"}


class _Checker:
    def __init__(self, tree, replace_one, delta_zero):
        self.tree, self.replace_one, self.delta_zero = tree, replace_one, delta_zero
        self.aliases = _aliases(tree)
        self.flags = []
        self.funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.summary = {name: self._summarize(f) for name, f in self.funcs.items()}
        self.has_change_one = bool(re.search(r"change_one|hamming|ChangeOne", ast.unparse(tree)))
        # lines near a vector-metric space (l1/l2/absolute distance): there d_in is a sensitivity, not a row count
        self.metric_lines = {getattr(n, "lineno", 0) for n in ast.walk(tree) if isinstance(n, ast.Call) and
                             (_attr(n) or "") in ("l1_distance", "l2_distance", "absolute_distance")}
        self.private_model = False   # any sign of a DP-trained model
        self.trainers = []           # (line, detail) of non-private training
        self.memo, self.ret_cache, self.active, self.returns = {}, {}, set(), []

    # ---- classification of calls -------------------------------------------------------------------------------
    def q(self, call):
        return _qual(call.func, self.aliases) or ""

    def is_noise(self, call):
        q = self.q(call)
        return (_attr(call) in NOISE_ATTRS or q.startswith(("diffprivlib.tools.", "diffprivlib.mechanisms."))
                or ("laplace" in q.lower() and q.endswith(".rvs")))

    def is_opendp_chain(self, node):
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                a = _attr(n) or (n.func.id if isinstance(n.func, ast.Name) else "")
                if a.startswith(("make_", "then_")) or a == "binary_search_chain":
                    return True
        return False

    def _summarize(self, fn):
        """noise: draws noise or calls a DP mechanism/model; trainer: fits a model with no noise;
        noisy_trainer: fits with noise (noisy GD, objective/output perturbation); chain: builds an OpenDP measurement."""
        qs = [(_attr(c), _qual(c.func, self.aliases) or "") for c in ast.walk(fn) if isinstance(c, ast.Call)]
        noise = any(a in NOISE_ATTRS or q.startswith(("diffprivlib.tools.", "diffprivlib.mechanisms.",
                                                         "diffprivlib.models.")) for a, q in qs)
        text = ast.unparse(fn)
        noise = noise or bool(re.search(r"then_laplace|make_laplace|then_gaussian|make_gaussian|then_geometric|"
                                        r"\.release\(\)|report_noisy", text))
        fit = bool(TRAIN_TEXT.search(text)) or any(q.startswith(SKLEARN_MODEL_MODULES) for _, q in qs)
        # trains with noise: a training loop that draws noise (noisy GD), or a DP-named helper that trains/calls a
        # trainer and draws noise (objective/output perturbation, noisy sufficient statistics)
        loops = [l for l in ast.walk(fn) if isinstance(l, (ast.For, ast.While))]
        noisy_loop = any(re.search(r"grad|backward|\.step\(|-=", ast.unparse(l)) and any(
            isinstance(c, ast.Call) and _attr(c) in NOISE_ATTRS for c in ast.walk(l)) for l in loops)
        dp_named = bool(re.search(r"(?i)dp|privat|nois|perturb|objective", fn.name))
        calls_helper = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and re.search(
            r"(?i)fit|train|logreg|ols|regress", c.func.id) for c in ast.walk(fn))
        noisy_trainer = noise and (noisy_loop or dp_named and (fit or calls_helper))
        return {"noise": noise, "trainer": fit and not noise, "noisy_trainer": noisy_trainer,
                "chain": self.is_opendp_chain(fn)}

    def flag(self, rule, node, detail):
        line = getattr(node, "lineno", 0)
        if not any(f["rule"] == rule and f["line"] == line for f in self.flags):
            self.flags.append({"rule": rule, "line": line, "detail": detail})

    @staticmethod
    def _derive(s):
        return (s - {"npw"}) | ({"npd"} if "npw" in s else set())

    # ---- labels of an expression -------------------------------------------------------------------------------
    def labels(self, e, env):
        """Memoised per statement (env does not change inside one statement)."""
        if e is None:
            return set()
        if id(e) not in self.memo:
            self.memo[id(e)] = self._labels(e, env)
        r = self.memo[id(e)]
        return [set(x) for x in r] if isinstance(r, list) else set(r)

    def flat(self, e, env):
        r = self.labels(e, env)
        return set().union(*r) if isinstance(r, list) else r

    def _labels(self, e, env):
        if isinstance(e, ast.Name):
            return set(env.get(e.id, ()))
        if isinstance(e, ast.Constant):
            return set()
        if isinstance(e, ast.Call):
            return self.call_labels(e, env)
        if isinstance(e, ast.Subscript):
            if isinstance(e.value, ast.Attribute) and e.value.attr == "shape":
                return {"gsize"} if "subset" in self.flat(e.value.value, env) else set()
            if isinstance(e.value, ast.Call) and _attr(e.value) == "value_counts":
                return {"gsize"}
            base = self.flat(e.value, env)
            sl = self.flat(e.slice, env)
            subset = any(isinstance(n, ast.Compare) for n in ast.walk(e.slice)) or "mask" in sl
            return (base - {"mask"}) | ({"subset"} if subset else set())
        if isinstance(e, ast.Attribute):
            base = self.flat(e.value, env)
            if e.attr in ("shape", "size"):
                return {"gsize"} if "subset" in base else set()
            if e.attr in ("index", "columns", "dtype", "dtypes", "ndim"):
                return set()
            return base
        if isinstance(e, ast.Compare):
            parts = set().union(*(self.flat(x, env) for x in [e.left, *e.comparators]))
            return self._derive(parts) - {"subset", "gsize", "scaled"} | {"mask"}
        if isinstance(e, ast.BinOp):
            left, right = self.flat(e.left, env), self.flat(e.right, env)
            noisy = ["noisy" in left or self.contains_noise(e.left, env),
                     "noisy" in right or self.contains_noise(e.right, env)]
            if isinstance(e.op, ast.Div) and noisy[0] and "gsize" in right and not noisy[1]:
                self.flag("group_size_public", e, f"`{ast.unparse(e.right)[:60]}` as the denominator of a noisy value")
            if isinstance(e.op, (ast.Add, ast.Sub)) and any(noisy) and not all(noisy):
                other = right if noisy[0] else left
                if "npd" in other:
                    self.flag("nonprivate_model", e, f"noise added to `{ast.unparse(e)[:60]}`, computed from a "
                                                     "non-private model")
                if "npw" in other:
                    self.private_model = True  # output perturbation of the weights
                return {"noisy"}
            if isinstance(e.op, (ast.BitAnd, ast.BitOr)):
                return left | right
            out = self._derive(left | right) - {"subset", "mask"}
            if isinstance(e.op, (ast.Mult, ast.Div)) and "noisy" in left | right:
                out -= {"gsize"}
            return out
        if isinstance(e, ast.Tuple) and isinstance(getattr(e, "ctx", None), ast.Load):
            return [self.flat(x, env) for x in e.elts]
        if isinstance(e, ast.Lambda):
            return {"measfn"} if self.is_opendp_chain(e) else set()
        out = set()
        for c in ast.iter_child_nodes(e):
            if isinstance(c, ast.expr):
                out |= self.flat(c, env)
        return self._derive(out) - {"subset", "scaler", "dpmodel", "meas", "measfn"}

    def contains_noise(self, e, env):
        for n in ast.walk(e):
            if isinstance(n, ast.Call) and (self.is_noise(n) or (isinstance(n.func, ast.Name) and (
                    self.summary.get(n.func.id, {}).get("noise") or "meas" in env.get(n.func.id, ())))):
                return True
        return False

    def args_of(self, call):
        return list(call.args) + [k.value for k in call.keywords]

    def scale_args(self, call, skip_first):
        """Arguments that set bounds/scale/sensitivity: keywords in SCALE_KWARGS, positional args after the first."""
        out = [k.value for k in call.keywords if k.arg in SCALE_KWARGS]
        return out + list(call.args[1 if skip_first else 0:])

    def check_scale(self, call, env, what, skip_first=True):
        for a in self.scale_args(call, skip_first):
            lab = self.flat(a, env)
            if "scaled" in lab:
                self.flag("private_bounds", call, f"`{ast.unparse(a)[:60]}` sets the bounds/scale of {what}")
            if "gsize" in lab and "noisy" not in lab:
                self.flag("group_size_public", call, f"`{ast.unparse(a)[:60]}` in {what}")

    def check_npd(self, call, env, what):
        for a in self.args_of(call):
            if "npd" in self.flat(a, env):
                self.flag("nonprivate_model", call, f"`{ast.unparse(a)[:60]}` passed to {what}, computed from a "
                                                    "non-private model")

    def call_labels(self, c, env):
        q, a = self.q(c), _attr(c)
        recv = self.flat(c.func.value, env) if isinstance(c.func, ast.Attribute) else set()
        kw = {k.arg for k in c.keywords}
        name = q.rsplit(".", 1)[-1]
        if q.startswith("diffprivlib.models."):
            self.private_model = True
            need = {"LogisticRegression": ["data_norm"], "LinearRegression": ["bounds_X", "bounds_y"],
                    "GaussianNB": ["bounds"]}.get(name, [])
            missing = [k for k in need if k not in kw]
            if missing:
                self.flag("missing_bounds", c, f"diffprivlib {name} (no {', '.join(missing)})")
            self.check_scale(c, env, f"diffprivlib {name}", skip_first=False)
            return {"dpmodel"}
        if q.startswith("diffprivlib.tools."):
            if name in DPL_NEED_BOUNDS and "bounds" not in kw:
                self.flag("missing_bounds", c, f"diffprivlib tools.{name}")
            if name in DPL_MEAN_LIKE and c.args and "subset" in self.flat(c.args[0], env):
                self.flag("group_size_public", c, f"tools.{name} on the subgroup `{ast.unparse(c.args[0])[:40]}` "
                                                  "(its length is treated as public)")
            if name.startswith("histogram") and self.replace_one:
                self.flag("add_remove_calibration", c, f"diffprivlib tools.{name} (sensitivity 1)")
            self.check_scale(c, env, f"tools.{name}")
            self.check_npd(c, env, f"tools.{name}")
            return {"noisy"}
        if self.is_noise(c):
            self.check_scale(c, env, "a noise draw")
            return {"noisy"}
        if isinstance(c.func, ast.Name) and "meas" in env.get(c.func.id, ()):
            self.check_npd(c, env, "an OpenDP measurement")
            return {"noisy"}
        if isinstance(c.func, ast.Name) and "measfn" in env.get(c.func.id, ()):
            return {"meas"}
        if a.startswith(("make_", "then_")) if a else False:
            for k in c.keywords:
                if k.arg == "bounds" and "scaled" in self.flat(k.value, env):
                    self.flag("private_bounds", c, f"`{ast.unparse(k.value)[:60]}` as OpenDP bounds")
            if a in ("make_clamp", "then_clamp") and c.args and "scaled" in self.flat(c.args[-1], env):
                self.flag("private_bounds", c, f"`{ast.unparse(c.args[-1])[:60]}` as OpenDP clamp bounds")
            return {"meas"}
        if a == "fit" and "dpmodel" in recv:
            for x in c.args[:1]:
                if "scaled" in self.flat(x, env):
                    self.flag("private_bounds", c, f"`{ast.unparse(x)[:60]}` (scaled with private statistics) "
                                                   "is the DP model's training data")
            return {"dpmodel"}
        if q.startswith(SKLEARN_MODEL_MODULES) or (q.startswith("torch.nn.") and name[:1].isupper()
                                                   and name not in ("Module", "Parameter")):
            self.trainers.append((c.lineno, f"{q.split('.')[0]} {name}"))
            return {"npw"}
        if q.startswith("sklearn.preprocessing.") and name in SCALERS:
            return {"scaler"} if name[:1].isupper() else {"scaled"}
        if a in ("fit_transform", "transform") and "scaler" in recv:
            return {"scaled"}
        if a in ("fit", "fit_transform") and "scaler" in recv:
            return {"scaler"}
        if a == "fit" and "npw" in recv:
            return {"npw"}
        if a in ("predict", "predict_proba", "decision_function", "score"):
            if "npw" in recv:
                return {"npd"}
            if "dpmodel" in recv:
                return set()
        if a == "value_counts" or (a in ("size", "count") and isinstance(c.func.value, ast.Call)
                                   and _attr(c.func.value) == "groupby"):
            return {"gsize"}
        if q == "len" and c.args:
            return {"gsize"} if "subset" in self.flat(c.args[0], env) else set()
        if q in ("numpy.sum", "numpy.count_nonzero") and c.args or (a == "sum" and not q.startswith("numpy.")):
            target = c.args[0] if q.startswith("numpy.") else c.func.value
            lab = self.flat(target, env)
            if "mask" in lab and "noisy" not in lab:
                return {"gsize"} | (lab & {"npd"})
        if a in STAT_ATTRS or (q.startswith("numpy.") and name in STAT_ATTRS):
            src = c.args[0] if q.startswith("numpy.") and c.args else (
                c.func.value if isinstance(c.func, ast.Attribute) else None)
            if src is not None and not isinstance(src, (ast.Constant, ast.List, ast.Tuple)):
                lab = self.flat(src, env)
                if "noisy" not in lab:
                    return {"scaled"} | (self._derive(lab) - {"subset", "mask", "gsize"})
        if isinstance(c.func, ast.Name) and c.func.id in self.funcs:
            return self.call_user(c, env)
        out = set(recv)
        for x in self.args_of(c):
            out |= self.flat(x, env)
        return self._derive(out) - {"subset", "gsize", "mask", "scaler", "dpmodel", "meas", "measfn"}

    def call_user(self, c, env):
        """Analyse the called function with the labels of its arguments; return the labels of its return value
        (a list for a tuple return)."""
        fn, s = self.funcs[c.func.id], self.summary[c.func.id]
        if s["noisy_trainer"]:
            self.private_model = True
            for x in self.args_of(c):
                if "scaled" in self.flat(x, env):
                    self.flag("private_bounds", c, f"`{ast.unparse(x)[:60]}` (scaled with private statistics) is "
                                                   f"the training data of `{fn.name}`")
        params = [p.arg for p in fn.args.args]
        penv = {p: self.flat(x, env) for p, x in zip(params, c.args)}
        penv.update({k.arg: self.flat(k.value, env) for k in c.keywords if k.arg in params})
        key = (fn.name, tuple(sorted((p, frozenset(v)) for p, v in penv.items())))
        if key in self.ret_cache:
            return self.ret_cache[key]
        if fn.name in self.active or len(self.active) > 6:
            return self._derive(set().union(*penv.values()) if penv else set())
        self.active.add(fn.name)
        self.returns.append([])
        memo = self.memo
        self.block(fn.body, dict(penv))
        self.memo = memo
        rets = self.returns.pop()
        self.active.discard(fn.name)
        tuples = [r for r in rets if isinstance(r, list)]
        if rets and len(tuples) == len(rets) and len({len(t) for t in tuples}) == 1:
            out = [set().union(*col) for col in zip(*tuples)]
        else:
            out = set().union(*[set().union(*r) if isinstance(r, list) else r for r in rets]) if rets else set()
        if s["trainer"]:
            self.trainers.append((c.lineno, f"`{fn.name}`"))
            ret_exprs = [n.value for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
            if isinstance(out, set) and ret_exprs and not out & {"npd", "noisy"}:
                out = out | {"npw"}  # a numpy/torch training helper returns the weights
            elif isinstance(out, list):
                elts = ret_exprs[-1].elts if ret_exprs and isinstance(ret_exprs[-1], ast.Tuple) else []
                out = [o | ({"npw"} if i < len(elts) and isinstance(elts[i], ast.Name)
                            and WEIGHT_NAME.match(elts[i].id) and not o & {"npd", "noisy"} else set())
                       for i, o in enumerate(out)]
        self.ret_cache[key] = out
        return out

    # ---- statements --------------------------------------------------------------------------------------------
    def assign(self, targets, value, env):
        lab = self.labels(value, env)
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.RShift) or (
                self.is_opendp_chain(value) and not self.contains_noise(value, env)):
            lab = {"meas"}
        for t in targets:
            if isinstance(t, (ast.Tuple, ast.List)) and isinstance(lab, list) and len(lab) == len(t.elts):
                for el, l in zip(t.elts, lab):
                    self._bind(el, l, env)
            else:
                self._bind(t, set().union(*lab) if isinstance(lab, list) else lab, env)

    def _bind(self, t, lab, env):
        if isinstance(t, ast.Name):
            env[t.id] = set(lab)
        elif isinstance(t, ast.Starred):
            self._bind(t.value, lab, env)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for el in t.elts:
                self._bind(el, lab, env)
        elif isinstance(t, (ast.Subscript, ast.Attribute)):  # x[...] = v / x.a = v: x keeps its labels and gains v's
            root = t
            while isinstance(root, (ast.Subscript, ast.Attribute)):
                root = root.value
            if isinstance(root, ast.Name):
                env[root.id] = env.get(root.id, set()) | (lab & {"scaled", "npw", "npd", "noisy"})

    def block(self, stmts, env):
        for s in stmts:
            self.stmt(s, env)

    def stmt(self, s, env):
        self.memo = {}
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return  # analysed per call (call_user) and once on its own (run)
        if isinstance(s, ast.Assign):
            self.assign(s.targets, s.value, env)
        elif isinstance(s, ast.AnnAssign) and s.value is not None:
            self.assign([s.target], s.value, env)
        elif isinstance(s, ast.AugAssign):
            lab = self.flat(s.value, env)
            for n in ast.walk(s.target):
                if isinstance(n, ast.Name):
                    if ("noisy" in lab or self.contains_noise(s.value, env)) and "npw" in env.get(n.id, ()):
                        self.private_model = True
                    env[n.id] = env.get(n.id, set()) | lab
                    break
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            self.assign([s.target], s.iter, env)
            self.block(s.body, env)
            self.block(s.orelse, env)
        elif isinstance(s, (ast.While, ast.If)):
            self.labels(s.test, env)
            self.block(s.body, env)
            self.block(s.orelse, env)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            self.block(s.body, env)
        elif isinstance(s, ast.Try):
            for b in (s.body, *[h.body for h in s.handlers], s.orelse, s.finalbody):
                self.block(b, env)
        elif isinstance(s, ast.Return) and s.value is not None:
            lab = self.labels(s.value, env)
            if self.returns:
                self.returns[-1].append(lab)
        elif isinstance(s, ast.Expr):
            lab = self.flat(s.value, env)
            c = s.value  # xs.append(v) / xs.extend(v) / d.update(v): xs gains v's labels
            if _attr(c) in ("append", "extend", "update", "insert") and isinstance(c.func.value, ast.Name):
                add = set().union(*(self.flat(x, env) for x in c.args)) if c.args else set()
                env[c.func.value.id] = env.get(c.func.value.id, set()) | (add & {"scaled", "npw", "npd", "noisy"})

    # ---- whole-script rules ------------------------------------------------------------------------------------
    def syntactic(self):
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call):
                kws = {k.arg: k.value for k in n.keywords}
                near_metric = any(abs(n.lineno - m) <= 6 for m in self.metric_lines)
                if self.replace_one and not self.has_change_one and not near_metric:
                    if "d_in" in kws and _const_num(kws["d_in"]) and kws["d_in"].value == 1:
                        self.flag("add_remove_calibration", n, "d_in=1")
                    if _attr(n) == "map" and n.args and _const_num(n.args[0]) and n.args[0].value == 1:
                        self.flag("add_remove_calibration", n, ".map(1)")
                    if "contributions" in kws and _const_num(kws["contributions"]) and kws["contributions"].value == 1:
                        self.flag("add_remove_calibration", n, "contributions=1")
                if self.replace_one and isinstance(n.func, ast.Name) and n.func.id == "max" and any(
                        re.search(r"eps", ast.unparse(x), re.I) for x in n.args):
                    self.flag("add_remove_calibration", n, f"`{ast.unparse(n)[:50]}` (parallel composition)")
                if self.delta_zero and "delta" in kws and _const_num(kws["delta"]) and kws["delta"].value > 0:
                    self.flag("positive_delta", n, f"delta={kws['delta'].value}")
            elif isinstance(n, ast.Assign) and self.delta_zero and _const_num(n.value) and n.value.value > 0:
                for t in n.targets:
                    if isinstance(t, ast.Name) and re.fullmatch(r"(?i)delta(_\w+)?|\w+_delta", t.id):
                        self.flag("positive_delta", n, f"{t.id} = {n.value.value}")
        # pure noise (a draw, possibly scaled; nothing added to it) put in a name that is never read in its function;
        # a noisy release (value + noise) that is only printed elsewhere is not this
        for scope in [self.tree, *self.funcs.values()]:
            nodes = list(ast.walk(scope)) if scope is not self.tree else [
                x for s in scope.body if not isinstance(s, (ast.FunctionDef, ast.ClassDef)) for x in ast.walk(s)]
            loads = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            for s in nodes:
                if isinstance(s, ast.Assign) and any(isinstance(x, ast.Call) and _attr(x) in (
                        "laplace", "gamma", "exponential", "gumbel") for x in ast.walk(s.value)) and not any(
                        isinstance(x, (ast.Add, ast.Sub)) for x in ast.walk(s.value)):
                    for t in s.targets:
                        if isinstance(t, ast.Name) and t.id not in loads and not t.id.startswith("_"):
                            self.flag("unused_noise", s, t.id)

    def run(self):
        self.syntactic()
        self.block(self.tree.body, {})
        for fn in self.funcs.values():  # functions on their own (also the ones never called)
            self.returns.append([])
            self.block(fn.body, {})
            self.returns.pop()
        if any(s["noisy_trainer"] for s in self.summary.values()):
            self.private_model = True
        if self.trainers and not self.private_model and not any(f["rule"] == "nonprivate_model" for f in self.flags):
            line, detail = min(self.trainers)
            self.flags.append({"rule": "nonprivate_model", "line": line,
                               "detail": f"{detail}; no DP-trained model found"})
        return self.flags


def output_flags(stdout: str, budget: float | None = 1.0) -> list[dict]:
    """Rules on the script's final JSON line: a DP value equal to its true value; epsilon_total over the budget."""
    obj = final_json(stdout)
    flags = []
    if not obj:
        return flags
    for k, v in obj.items():  # compared with true_*, true_original_* and true_own_*
        if k.startswith("dp_") and v not in (None, [], {}) and any(
                obj.get(f"true{p}_{k[3:]}", object()) == v for p in ("", "_original", "_own")):
            flags.append({"rule": "dp_equals_true", "line": 0, "detail": f"`{k}`"})
    eps = obj.get("epsilon_total")
    if budget is not None and isinstance(eps, (int, float)) and eps > budget * (1 + 1e-3):
        flags.append({"rule": "over_budget", "line": 0, "detail": f"{eps:g}"})
    return flags


def final_json(stdout: str):
    """The last output line that parses as a JSON object, else None."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                return obj if isinstance(obj, dict) else None
            except ValueError:
                return None
    return None


def check(code: str, stdout: str | None = None, budget: float | None = 1.0, delta_zero: bool = True,
          replace_one: bool = True) -> list[dict]:
    """All flags for a script, sorted by rule and line. stdout adds the final-JSON rules; budget, delta_zero and
    replace_one are the task's privacy setup. Code that does not parse gets no flags."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    flags = _Checker(tree, replace_one, delta_zero).run()
    if stdout is not None:
        flags += output_flags(stdout, budget)
    for f in flags:
        f["message"] = MESSAGES[f["rule"]].format(detail=f["detail"], budget=budget or 0)
    return sorted(flags, key=lambda f: (f["rule"], f["line"]))


def feedback(flags: list[dict]) -> str:
    """One paragraph per rule (first line numbers listed), for the repair prompt."""
    out = []
    for rule in dict.fromkeys(f["rule"] for f in flags):
        fs = [f for f in flags if f["rule"] == rule]
        lines = ", ".join(str(f["line"]) for f in fs if f["line"])
        out.append(f"- {fs[0]['message']}" + (f" (line {lines})" if lines else ""))
    return "\n".join(out)
