#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  benchmark/detect_embedding_profile.sh \
    --endpoint <url> \
    --model <model_id> \
    [--api-key <key>] \
    [--probe-text <text>] \
    [--output summary|json|dimension]

Outputs a quick profile of one real embedding response and infers style hints.
USAGE
}

ENDPOINT=""
MODEL=""
API_KEY=""
PROBE_TEXT="Munind embedding profile probe text."
OUTPUT="summary"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)
      ENDPOINT="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --api-key)
      API_KEY="${2:-}"
      shift 2
      ;;
    --probe-text)
      PROBE_TEXT="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${ENDPOINT}" || -z "${MODEL}" ]]; then
  echo "--endpoint and --model are required" >&2
  usage >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

payload="$(jq -cn --arg model "${MODEL}" --arg input "${PROBE_TEXT}" '{model: $model, input: $input}')"

headers=(-H "Content-Type: application/json")
if [[ -n "${API_KEY}" ]]; then
  headers+=(-H "Authorization: Bearer ${API_KEY}")
fi

response="$(curl -sS -X POST "${ENDPOINT}" "${headers[@]}" -d "${payload}")"

if ! profile_json="$(jq -ce '
  .data[0].embedding as $e
  | if ($e | type) != "array" then
      error("response missing data[0].embedding array")
    else
      ($e | length) as $n
      | {
          endpoint_model: (.model // null),
          dimension: $n,
          l2_norm: (if $n == 0 then 0 else (($e | map(. * .) | add) | sqrt) end),
          mean: (if $n == 0 then 0 else (($e | add) / $n) end),
          mean_abs: (if $n == 0 then 0 else (($e | map(if . < 0 then -. else . end) | add) / $n) end),
          min: (if $n == 0 then 0 else ($e | min) end),
          max: (if $n == 0 then 0 else ($e | max) end)
        }
    end
' <<<"${response}")"; then
  echo "Failed to parse embedding response from ${ENDPOINT}" >&2
  echo "Response (first 400 chars):" >&2
  printf '%s\n' "${response}" | head -c 400 >&2
  echo >&2
  exit 1
fi

style="$(jq -r '
  if .dimension == 0 then
    "empty embedding"
  elif (.l2_norm >= 0.98 and .l2_norm <= 1.02) then
    "likely L2-normalized dense vector"
  else
    "dense vector (not unit-normalized)"
  end
' <<<"${profile_json}")"

case "${OUTPUT}" in
  dimension)
    jq -r '.dimension' <<<"${profile_json}"
    ;;
  json)
    jq --arg style "${style}" '. + {style_hint: $style}' <<<"${profile_json}"
    ;;
  summary)
    echo "Embedding profile"
    echo "  endpoint: ${ENDPOINT}"
    echo "  requested_model: ${MODEL}"
    echo "  response_model: $(jq -r '.endpoint_model // "(missing)"' <<<"${profile_json}")"
    echo "  dimension: $(jq -r '.dimension' <<<"${profile_json}")"
    echo "  l2_norm: $(jq -r '.l2_norm' <<<"${profile_json}")"
    echo "  mean: $(jq -r '.mean' <<<"${profile_json}")"
    echo "  mean_abs: $(jq -r '.mean_abs' <<<"${profile_json}")"
    echo "  min: $(jq -r '.min' <<<"${profile_json}")"
    echo "  max: $(jq -r '.max' <<<"${profile_json}")"
    echo "  style_hint: ${style}"
    ;;
  *)
    echo "Invalid --output value: ${OUTPUT}" >&2
    exit 1
    ;;
esac
