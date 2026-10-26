#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_root="${SIRDET_DATA_ROOT:-${LERDET_DATA_ROOT:-${root}/test_data}}"
manifest="${data_root}/manifest.csv"
model="${SIRDET_MODEL:-${LERDET_MODEL:-}}"
[[ -s "${manifest}" ]] || { echo "Missing dataset manifest: ${manifest}" >&2; exit 1; }
[[ -s "${model}" ]] || { echo "Missing RKNN model. Set SIRDET_MODEL to the .rknn file." >&2; exit 1; }
name="${1:-$(awk -F, 'NR == 2 {print $1}' "${manifest}")}"
report="${root}/outputs/rknn_perf_probe.txt"

mkdir -p "${root}/outputs"
"${root}/bin/lerdet_perf_probe" \
    "${model}" \
    "${data_root}/visible/${name}.jpg" \
    "${data_root}/infrared/${name}.jpg" \
    "${report}"
echo "Report: ${report}"
