#!/usr/bin/env bash
# Set up a SPHERE experiment node to run generated code (see sphere/README.md). Idempotent; run as the
# experiment user (passwordless sudo) after every activation: the iptables rule does not survive a reboot.
#   bash sphere/setup_node.sh            (from the repo copy on the node)
set -euo pipefail
SANDBOX_USER=${SANDBOX_USER:-dpsandbox}

# 1. The 200 GB extra disk at /data (formatted only if it has no filesystem; /data is wiped at deactivation).
if ! mountpoint -q /data; then
  sudo blkid /dev/vdb >/dev/null || sudo mkfs.ext4 -q -F /dev/vdb
  sudo mkdir -p /data && sudo mount /dev/vdb /data && sudo chown "$(id -u):$(id -g)" /data
fi

# 2. An unprivileged user that runs generated code: no sudo, no login, no keys, no access to your files.
id "$SANDBOX_USER" >/dev/null 2>&1 || sudo useradd --system --shell /usr/sbin/nologin \
  --home-dir /data/sandbox_home --create-home "$SANDBOX_USER"
chmod 700 "$HOME"
# The repo and the app data: yours only (the README moves the repo to /data/llmdpeval after this script and runs
# the same chmod; running this script again after the move also covers it).
for d in /data/llmdpeval /data/apps; do
  if [ -d "$d" ]; then chmod 750 "$d"; fi
done

# 3. No network for that user (IPv4 and IPv6). Loopback stays usable, except DNS: a local resolver
#    (127.0.0.53) would forward queries, and a query name can carry data out.
for ipt in iptables ip6tables; do
  sudo "$ipt" -C OUTPUT -o lo -m owner --uid-owner "$SANDBOX_USER" -j ACCEPT 2>/dev/null ||
    sudo "$ipt" -I OUTPUT 1 -o lo -m owner --uid-owner "$SANDBOX_USER" -j ACCEPT
  for proto in udp tcp; do  # inserted above the loopback ACCEPT
    sudo "$ipt" -C OUTPUT -p "$proto" --dport 53 -m owner --uid-owner "$SANDBOX_USER" -j REJECT 2>/dev/null ||
      sudo "$ipt" -I OUTPUT 1 -p "$proto" --dport 53 -m owner --uid-owner "$SANDBOX_USER" -j REJECT
  done
  sudo "$ipt" -C OUTPUT -m owner --uid-owner "$SANDBOX_USER" -j REJECT 2>/dev/null ||
    sudo "$ipt" -A OUTPUT -m owner --uid-owner "$SANDBOX_USER" -j REJECT
done

# 4. Self-check: the sandbox user cannot reach the internet, resolve names, or read your home and repo.
if sudo -n -u "$SANDBOX_USER" timeout 10 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null; then
  echo "FAIL: $SANDBOX_USER has network"; exit 1; fi
if sudo -n -u "$SANDBOX_USER" timeout 10 getent ahosts example.com >/dev/null 2>&1; then
  echo "FAIL: $SANDBOX_USER can resolve names (if /etc/nsswitch.conf lists 'resolve', queries go to"
  echo "      systemd-resolved over a local socket, which iptables does not see: remove 'resolve' there)"; exit 1; fi
if sudo -n -u "$SANDBOX_USER" ls "$HOME" >/dev/null 2>&1; then echo "FAIL: $SANDBOX_USER can read $HOME"; exit 1; fi
if [ -d /data/llmdpeval ] && sudo -n -u "$SANDBOX_USER" ls /data/llmdpeval >/dev/null 2>&1; then
  echo "FAIL: $SANDBOX_USER can read /data/llmdpeval"; exit 1; fi
echo "node ready: $SANDBOX_USER has no network or DNS and no access to $HOME or /data/llmdpeval"
echo "run evaluations with: export LLMDPEVAL_SANDBOX_USER=$SANDBOX_USER"
