#!/usr/bin/env bash
# Build the arXiv upload (arxiv/ and cve-paper-arxiv.tar.gz) from paper/.
# arXiv does not run BibTeX, so the compiled main.bbl is included.
set -euo pipefail
cd "$(dirname "$0")/.."
(cd paper && tectonic -X compile main.tex >/dev/null)
rm -rf arxiv && mkdir -p arxiv/figures
cp paper/main.tex paper/numbers.tex paper/refs.bib paper/main.bbl arxiv/
cp -R paper/sections paper/tables arxiv/
cp paper/figures/*.pdf arxiv/figures/
tar -czf cve-paper-arxiv.tar.gz -C arxiv .
echo "wrote cve-paper-arxiv.tar.gz ($(du -h cve-paper-arxiv.tar.gz | cut -f1))"
