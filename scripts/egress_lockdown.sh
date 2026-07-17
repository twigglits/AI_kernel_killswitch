#!/usr/bin/env bash
# Egress lockdown for the serving box (optional hardening, root).
#
# A runtime killswitch cannot reach a copy of the model that has already left
# the machine, so the strongest pairing is to stop weights leaving in the
# first place. This script installs a host-wide default-deny on NEW OUTBOUND
# connections using nftables:
#
#   - loopback traffic is allowed (the shred-helper socket, local tooling);
#   - replies to INBOUND connections are allowed (conntrack), so the operator
#     API on :8000 and SSH keep working;
#   - everything else outbound — including DNS — is dropped and counted.
#
# Run it AFTER provisioning (fetch_checkpoint.py needs network to download the
# model). Allow specific destinations, e.g. a metrics host, with:
#
#   KS_EGRESS_ALLOW="192.0.2.10/32 198.51.100.0/24" egress_lockdown.sh on
#
# Usage:  egress_lockdown.sh on|off|status
set -euo pipefail

TABLE=ks_egress

[ "$(id -u)" -eq 0 ] || { echo "must run as root" >&2; exit 1; }

case "${1:-on}" in
  on)
    allow_rules=""
    for cidr in ${KS_EGRESS_ALLOW:-}; do
      case "$cidr" in
        *:*) allow_rules+="        ip6 daddr $cidr accept"$'\n' ;;
        *)   allow_rules+="        ip daddr $cidr accept"$'\n' ;;
      esac
    done
    nft -f - <<EOF
table inet $TABLE
delete table inet $TABLE
table inet $TABLE {
    chain output {
        type filter hook output priority 0; policy drop;
        oif "lo" accept
        ct state established,related accept
$allow_rules        counter comment "ks-egress dropped"
    }
}
EOF
    echo "egress lockdown ACTIVE: new outbound connections are dropped."
    echo "allowed: loopback, replies to inbound${KS_EGRESS_ALLOW:+, $KS_EGRESS_ALLOW}"
    echo "inspect drops with: $0 status"
    ;;
  off)
    nft delete table inet "$TABLE" 2>/dev/null || true
    echo "egress lockdown removed."
    ;;
  status)
    nft list table inet "$TABLE" 2>/dev/null || echo "egress lockdown not active."
    ;;
  *)
    echo "usage: $0 on|off|status" >&2; exit 2
    ;;
esac
