#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${HERE}/../../.." && pwd)"
ACLIB_PYTHON="/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python"
SMAC_SITE_PACKAGES="/home/io632776/work/py-envs/py3.12-smac/lib/python3.9/site-packages"
EPM_SOURCE="${REPOSITORY_ROOT}/external/aclib-surrogates/epm"

if [[ ! -x "${ACLIB_PYTHON}" ]]; then
    echo "Missing ACLib Python interpreter: ${ACLIB_PYTHON}" >&2
    exit 1
fi
if [[ ! -d "${SMAC_SITE_PACKAGES}" ]]; then
    echo "Missing SMAC site-packages directory: ${SMAC_SITE_PACKAGES}" >&2
    exit 1
fi

export PYTHONPATH="${SMAC_SITE_PACKAGES}:${EPM_SOURCE}:${HERE}:${PYTHONPATH:-}"
exec "${ACLIB_PYTHON}" "$@"

