#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: install_skills.sh DESTINATION" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
source_root="$repo_root/skills"
destination_root="$1"
skill_names=(
  nanoscigpt-research-baseline-builder
  nanogpt-pretraining
  nanoscigpt-scientific-language
  autoresearch-model-iteration
  ai-scientist-v1-workflow
  ai-scientist-v2-tree-search
)

for name in "${skill_names[@]}"; do
  source="$source_root/$name"
  target="$destination_root/$name"
  if [[ ! -f "$source/SKILL.md" ]]; then
    echo "source skill is incomplete: $source" >&2
    exit 2
  fi
  if [[ -e "$target" ]]; then
    echo "skill already exists: $target" >&2
    exit 2
  fi
done

mkdir -p -- "$destination_root"
for name in "${skill_names[@]}"; do
  cp -R -- "$source_root/$name" "$destination_root/$name"
done

echo "installed 6 nanoSciGPT course skills -> $destination_root"
