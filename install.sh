#!/bin/sh
# Install or update katzran on Debian 13 with an Intel iGPU (run as root inside the machine, from this checkout).
# Models come from this repo's release (built by the models workflow) and are checked against checksums/models.sha256.
set -eu
cd "$(dirname "$0")"
VARIANT=$(sed -n "s/^VARIANT=//p" /etc/katzran/katzran.env 2>/dev/null || true); VARIANT=${VARIANT:-int8}
RELEASE=https://github.com/kolmanv/katzran/releases/download/$(cat models/RELEASE)
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq python3-venv ca-certificates curl >/dev/null

# Intel GPU compute runtime (Debian 13 doesn't ship it): pinned release, checked against checksums/intel-gpu.sha256
if ! dpkg -s intel-opencl-icd >/dev/null 2>&1; then
  apt-get install -y -qq ocl-icd-libopencl1 >/dev/null
  d=$(mktemp -d)
  while read -r sum name url; do curl -fsSL -o "$d/$name" "$url"; done < checksums/intel-gpu.sha256
  awk '{print $1"  "$2}' checksums/intel-gpu.sha256 | (cd "$d" && sha256sum -c --quiet)
  apt-get install -y -qq "$d"/*.deb >/dev/null; rm -r "$d"
fi

id katzran >/dev/null 2>&1 || useradd -r -d /opt/katzran -s /usr/sbin/nologin -G render katzran
install -d -o katzran -g katzran /opt/katzran /opt/katzran/models /var/cache/katzran
install -d -m 750 -g katzran /etc/katzran
[ -f /etc/katzran/tokens ] || install -m 640 -g katzran /dev/null /etc/katzran/tokens
[ -f /etc/katzran/katzran.env ] || install -m 640 -g katzran katzran.env.example /etc/katzran/katzran.env
install -m 644 katzran.py /opt/katzran/katzran.py
[ -d /opt/katzran/venv ] || python3 -m venv /opt/katzran/venv
/opt/katzran/venv/bin/pip install -q -r requirements.txt

M=/opt/katzran/models
for name in he-turbo general-turbo; do
  f=$name-$VARIANT-ov.tar
  [ -d "$M/$name-$VARIANT-ov" ] && continue
  curl -fsSL -o "$M/$f" "$RELEASE/$f"
  grep " $f\$" checksums/models.sha256 | (cd $M && sha256sum -c --quiet) || { rm "$M/$f"; exit 1; }
  tar -xf "$M/$f" -C $M && rm "$M/$f"
done
[ -d $M/detector-small ] || /opt/katzran/venv/bin/python -c 'from huggingface_hub import snapshot_download as s
s("Systran/faster-whisper-small", revision="536b0662742c02347bc0e980a01041f333bce120", local_dir="/opt/katzran/models/detector-small")'
chown -R katzran:katzran $M

install -m 644 katzran.service /etc/systemd/system/katzran.service
systemctl daemon-reload && systemctl enable -q katzran && systemctl restart katzran
echo "katzran installed ($VARIANT). New caller token: /opt/katzran/venv/bin/python /opt/katzran/katzran.py token <name>"
