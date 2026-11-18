# Running evaluations on a SPHERE node

An XDC is a small gateway (about 2 GB, 2 CPUs) and holds your API keys, so generated code never runs there.
Evaluation runs on an experiment node, as a separate user with no network.

## Get a node (web UI, https://launch.sphere-testbed.net, project `teppit`)
1. Experiments -> Manage -> New Experiment (e.g. `dpeval`).
2. Model Editor -> paste `sphere/model.py` -> Compile Code -> Push -> your experiment.
3. Revision page -> Reserve Revision (up to 30 days) -> Reservations -> Actions -> Activate.
4. XDCs -> your XDC -> Actions -> Attach -> the activation. Then `ssh <xdc>` and `ssh eval`.

## Set up the node (once per activation)
```bash
# from the XDC: copy the repo and micromamba to the node (the node's home is not shared with the XDC)
git -C ~/llmdpeval archive HEAD | ssh eval 'mkdir -p /tmp/repo && tar xf - -C /tmp/repo'
ssh eval 'bash /tmp/repo/sphere/setup_node.sh &&   # disk, sandbox user, no network or DNS
  mv /tmp/repo /data/llmdpeval && chmod 750 /data/llmdpeval'   # the repo is not readable by the sandbox user
tar cf - -C ~/bin micromamba | ssh eval 'mkdir -p /data/bin && tar xf - -C /data/bin'
ssh eval '/data/bin/micromamba create -y -p /data/envs/py311 -c conda-forge python=3.11 &&
  /data/envs/py311/bin/python -m pip install -r /data/llmdpeval/requirements-lock.txt &&
  /data/envs/py311/bin/python -m pip install -e /data/llmdpeval --no-deps'
```
App data under `applications/*/data` is not in git: copy it to the node too. Then run
`bash /data/llmdpeval/sphere/setup_node.sh` once more: it checks that the sandbox user cannot read `/data/llmdpeval`.

## The three steps
1. **Generate on the XDC** (calls to sphere-llm.net only, no code runs; `pipeline.toml` has `repair = 0`):
   `python -m llmdpeval.rag build` once, then `python -m llmdpeval.run_grid --grid-dir <grid>`. Copy the grid to
   the node.
2. **Evaluate on the node** (below): `run_grid --evaluate` writes `<grid>/results.jsonl`.
3. **Repair on the node**, 2 rounds, then evaluate the repaired copy:
   `python -m llmdpeval.repair <grid> <grid>_fix2 --rounds 2` and `run_grid --grid-dir <grid>_fix2 --evaluate`.
   Repair calls the model again, so the node needs the sphere key (`~/.llmdpeval_keys/sphere_key.txt`; the home
   folder is mode 700) and the built index (`storage/`, copy it from the XDC or run `rag build` on the node).

## Evaluate
```bash
cd /data/llmdpeval
export LLMDPEVAL_SANDBOX_USER=dpsandbox           # required on Linux; generated code runs as this user
export LLMDPEVAL_RUN_TIMEOUT=120                  # seconds per script (raise for big apps)
export LLMDPEVAL_SANDBOX_MEMORY_GB=16             # memory limit per script
/data/envs/py311/bin/python -m llmdpeval.sandbox  # self-check: no network, no access to your home, no leftovers
/data/envs/py311/bin/python -m llmdpeval.run_grid --grid-dir <grid> --evaluate --eval-workers 8 ...
/data/envs/py311/bin/python -m llmdpeval.repair <grid> <grid>_fix2 --rounds 2
/data/envs/py311/bin/python -m llmdpeval.run_grid --grid-dir <grid>_fix2 --evaluate --eval-workers 8
```
Each script runs in its own private scratch folder (owned by the sandbox user, mode 700) and its own process
group, which is killed when the script ends or times out. A process that leaves its group is killed at the end of
the pass (`pkill -u dpsandbox` after `--evaluate` and `grade median`, after each script in `repair`), so do not run
two evaluation passes on one node at the same time.

## Limits
- `/data` is wiped when the activation is deactivated: copy results back to the XDC home first.
- The iptables rule is lost on reboot: run `setup_node.sh` again after any restart.
- The sandbox user can still use CPU and disk on the node; it cannot reach the network, resolve names or read
  your files.
