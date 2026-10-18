#!/usr/bin/env bash
# PostgreSQL 18 with pgvector on Windows without Docker or an installer: everything lives in .tmp/.
#   setup  download the EDB binaries and the pgvector source (pinned SHA-256), extract, build pgvector
#          with MSVC (Visual Studio Build Tools) into the local copy
#   start  initdb on first use, then start on 127.0.0.1:54329 (trust auth, loopback only)
#   stop   stop the server
# CI (Linux) uses the pgvector/pgvector container instead; see .github/workflows/ci.yml.
set -euo pipefail
cd "$(dirname "$0")/.."

PG_ZIP=postgresql-18.6-1-windows-x64-binaries.zip
PG_URL=https://get.enterprisedb.com/postgresql/$PG_ZIP
PG_SHA=fbe23da234ee31547bf8a36d29dfd81e82b849df2d2b78d2eecb43d360252f8c
PGV=0.8.7
PGV_URL=https://github.com/pgvector/pgvector/archive/refs/tags/v$PGV.tar.gz
PGV_SHA=cac0b10c360f05b2d521200105ba3697e773d4cd3731f5a915a7e37ebe0bea85
VCVARS=${VCVARS:-"C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"}
PORT=${PGPORT:-54329}
T=.tmp
PG=$T/pg-extract/pgsql
DATA=$T/pgdata

# fetch downloads a file once and checks its SHA-256.
fetch() {
  local url=$1 out=$2 sha=$3
  mkdir -p "$T/downloads"
  [ -f "$out" ] || curl -fsSL --proto '=https' -o "$out.part" "$url" && { [ -f "$out" ] || mv "$out.part" "$out"; }
  echo "$sha  $out" | sha256sum -c - >/dev/null || { echo "checksum mismatch: $out" >&2; rm -f "$out"; exit 1; }
}

setup() {
  fetch "$PG_URL" "$T/downloads/$PG_ZIP" "$PG_SHA"
  fetch "$PGV_URL" "$T/downloads/pgvector-$PGV.tar.gz" "$PGV_SHA"
  if [ ! -x "$PG/bin/postgres.exe" ]; then
    python -I - "$T/downloads/$PG_ZIP" "$T/pg-extract" <<'EOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
keep = [n for n in z.namelist() if n.split("/")[1:2] and n.split("/")[1] in ("bin", "lib", "share", "include")]
z.extractall(sys.argv[2], members=keep)
EOF
  fi
  if [ ! -f "$PG/lib/vector.dll" ]; then
    mkdir -p "$T/src" && tar -xzf "$T/downloads/pgvector-$PGV.tar.gz" -C "$T/src"
    local r; r=$(pwd -W | sed 's|/|\\|g')
    cat > "$T/build-pgvector.cmd" <<EOF
@echo off
set VSCMD_SKIP_SENDTELEMETRY=1
set TMP=$r\\.tmp
set TEMP=$r\\.tmp
set LOCALAPPDATA=$r\\.tmp\\localappdata
set APPDATA=$r\\.tmp\\appdata
set PATH=C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer;%PATH%
call "$VCVARS" >nul || exit /b 1
set PGROOT=$r\\.tmp\\pg-extract\\pgsql
cd /d $r\\.tmp\\src\\pgvector-$PGV || exit /b 1
nmake /nologo /F Makefile.win || exit /b 1
nmake /nologo /F Makefile.win install || exit /b 1
EOF
    cmd //c "$(cygpath -w "$T/build-pgvector.cmd")" > "$T/build-pgvector.log" 2>&1 || { tail -20 "$T/build-pgvector.log"; exit 1; }
  fi
  echo "postgres $("$PG/bin/postgres.exe" --version | awk '{print $3}') with pgvector $PGV in $PG"
}

start() {
  [ -x "$PG/bin/postgres.exe" ] || { echo "run: scripts/pg-local.sh setup" >&2; exit 1; }
  if [ ! -f "$DATA/PG_VERSION" ]; then
    "$PG/bin/initdb.exe" -D "$DATA" -U postgres --auth=trust -E UTF8 --locale=C > "$T/initdb.log"
  fi
  "$PG/bin/pg_ctl.exe" -D "$DATA" -l "$T/pg.log" -w \
    -o "-p $PORT -c listen_addresses=127.0.0.1 -c shared_buffers=64MB -c max_connections=20" start \
    > "$T/pg_ctl.log" 2>&1 < /dev/null  # the server inherits these handles; a pipe here would never close
  "$PG/bin/psql.exe" -h 127.0.0.1 -p "$PORT" -U postgres -qAt -c "SELECT 1 FROM pg_database WHERE datname='hsearch'" | grep -q 1 \
    || "$PG/bin/createdb.exe" -h 127.0.0.1 -p "$PORT" -U postgres hsearch
  echo "HSEARCH_DSN=postgres://postgres@127.0.0.1:$PORT/hsearch?sslmode=disable"
}

stop() {
  [ -f "$DATA/postmaster.pid" ] && "$PG/bin/pg_ctl.exe" -D "$DATA" -m fast -w stop || echo "not running"
}

case "${1:-}" in
  setup) setup ;;
  start) start ;;
  stop) stop ;;
  *) echo "usage: scripts/pg-local.sh setup|start|stop" >&2; exit 2 ;;
esac
