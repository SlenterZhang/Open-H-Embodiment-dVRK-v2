#!/usr/bin/env bash
# Stream the stereo endoscope to an Apple Vision Pro: open https://<this-pc-ip>:8443 in its Safari.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IP="${AVP_HOST_IP:-$(hostname -I | awk '{print $1}')}"
CERT_DIR="$ROOT/.certs"
if [[ ! -f "$CERT_DIR/avp.crt" ]] || ! openssl x509 -in "$CERT_DIR/avp.crt" -noout -text | grep -q "IP Address:$IP"; then
  mkdir -p "$CERT_DIR"
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=dvrk-avp" \
    -addext "subjectAltName=IP:$IP" \
    -keyout "$CERT_DIR/avp.key" -out "$CERT_DIR/avp.crt" 2>/dev/null
  echo "created self-signed certificate for $IP in $CERT_DIR"
fi

echo "open https://$IP:8443 in Safari on the Vision Pro"
exec uv run --extra avp dvrk-avp --cert "$CERT_DIR/avp.crt" --key "$CERT_DIR/avp.key" "$@"
