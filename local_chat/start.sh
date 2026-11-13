#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export DSH_HOME="${DSH_HOME:-$HOME/.local/share/dsh-gum}"
export LOCAL_GUM_API_KEY="${LOCAL_GUM_API_KEY:-local-gum-key}"
mkdir -p "$DSH_HOME/profiles/web"
if [[ ! -f "$DSH_HOME/profiles/web/cordis.patch.yml" ]]; then
    cp "$root/local_chat/cordis.patch.yml" "$DSH_HOME/profiles/web/cordis.patch.yml"
fi

# The API runs under systemd (gum-openai.service, LimitMEMLOCK=infinity). Only fall
# back to an in-process API when the service is not managing it.
gum_api_pid=""
cleanup() {
    if [ -n "$gum_api_pid" ]; then
        kill "$gum_api_pid" 2>/dev/null || true
        wait "$gum_api_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if curl --silent --fail http://127.0.0.1:8765/health >/dev/null 2>&1; then
    echo "GUM API already served by gum-openai.service"
elif ! systemctl is-active --quiet gum-openai.service 2>/dev/null; then
    python3 "$root/local_chat/gum_openai_server.py" &
    gum_api_pid=$!
    for _ in $(seq 1 60); do
        if curl --silent --fail http://127.0.0.1:8765/health >/dev/null; then break; fi
        if ! kill -0 "$gum_api_pid" 2>/dev/null; then break; fi
        sleep 1
    done
    if ! kill -0 "$gum_api_pid" 2>/dev/null || ! curl --silent --fail http://127.0.0.1:8765/health >/dev/null; then
        echo "GUM local API did not start; see the Python error above." >&2
        exit 1
    fi
fi

npx --offline --yes @deepseek-ai/dsh@0.2.2-rc.1 web --no-open "$@"
