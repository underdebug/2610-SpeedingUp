#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

code --uninstall-extension local.mem

vsix="mem-$(node -p 'require("./package.json").version').vsix"
vsce package --no-dependencies --allow-missing-repository --skip-license --out "$vsix"
code --install-extension "$vsix" --force

