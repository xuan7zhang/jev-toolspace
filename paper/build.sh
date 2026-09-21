#!/bin/bash
# local build (Overleaf builds main.tex directly with the Optimistic font)
cd "$(dirname "$0")"; J='\def\tealocal{1}\input{main}'
pdflatex -interaction=nonstopmode -jobname=main "$J" >/dev/null; bibtex main >/dev/null
pdflatex -interaction=nonstopmode -jobname=main "$J" >/dev/null; pdflatex -interaction=nonstopmode -jobname=main "$J" >/dev/null
grep -c "^!" main.log
