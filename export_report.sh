#!/bin/sh
# Export the report: every notebook as static HTML into report/, then zip it for email.
# The HTML files link to each other by name, so keep them together in one folder.
# Why main.html has no code: it is the report, read for the results. The companion notebooks
# keep their code, since that is where a quant reader checks how each step was done.
set -e
cd "$(dirname "$0")"
mkdir -p report
echo "exporting main"
(cd src && uv run marimo export html --no-include-code main.py -o ../report/main.html)
for nb in data_nb factors_nb ols_nb estimation_nb validation_nb hypotheses_nb; do
    echo "exporting $nb"
    (cd src && uv run marimo export html "$nb.py" -o "../report/$nb.html")
done
rm -f report.zip
zip -rq report.zip report
echo "done: report/ and report.zip"
