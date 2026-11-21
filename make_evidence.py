"""Generate a labelled local result HTML summary from actual evaluation files."""

import csv
import html
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "artifacts"


def main():
    with (EVIDENCE / "hyperparameter_comparison.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    fields = ["configuration", "accuracy", "precision", "recall", "f1", "auc", "malignant_recall", "false_benign_count"]
    headers = ["Configuration", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Malignant recall", "False benign"]
    def cell(row, key):
        value = row[key]
        if key in ("configuration", "false_benign_count"):
            return f"<td>{html.escape(value)}</td>"
        return f"<td>{float(value):.4f}</td>"

    cells = "".join(
        "<tr>" + "".join(cell(row, key) for key in fields) + "</tr>"
        for row in records
    )
    heading = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>LOCAL execution evidence - SageMaker assignment</title>
<style>
body{{font:17px/1.5 system-ui,Arial;color:#172a40;background:#f1f5f9;margin:0}}
main{{max-width:1280px;margin:28px auto;padding:28px 36px;background:white;border-radius:20px;box-shadow:0 8px 30px #cbd5e1}}
h1{{font-size:31px;margin:0 0 8px}} .tag{{display:inline-block;background:#fff1b7;border:2px solid #c58909;border-radius:7px;padding:7px 14px;font-weight:800;letter-spacing:.07em}}
.sub{{color:#4b6077}} table{{border-collapse:collapse;width:100%;font-size:14px;margin-top:26px}}
td,th{{padding:12px 9px;border:1px solid #dce4ec;text-align:center}} th{{background:#11375e;color:white}} tr:nth-child(even){{background:#f5f9ff}}
.plots{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:24px}} img{{width:100%;border:1px solid #d4dfeb;border-radius:8px}}
.note{{background:#e7f2fd;padding:16px;border-radius:9px;border-left:5px solid #1279c6;margin-top:23px}}
</style></head><body><main>
<div class="tag">VERIFIED LOCAL EXECUTION - NOT AN AWS CONSOLE SCREENSHOT</div>
<h1>Breast Cancer Random Forest - Assignment Evaluation</h1>
<div class="sub">Wisconsin Diagnostic (569 rows, 30 features) | Stratified split: 455 train / 114 test | Seed 42 | Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>
<table><thead><tr>{heading}</tr></thead><tbody>{cells}</tbody></table>
<div class="note">The model is for coursework only. Positive class = benign (1). Medical screening prioritises avoiding false benign decisions; both runs produced three false benign results on the test set.</div>
<div class="plots"><img alt="Confusion matrices" src="confusion_matrices.png"><img alt="ROC curves" src="roc_curves.png"></div>
<p class="sub">These results come from two actual local training runs (`python evaluate_local.py`). AWS training and endpoint metrics must be measured separately in the user's account.</p>
</main></body></html>"""
    path = EVIDENCE / "local_evidence.html"
    path.write_text(page, encoding="utf-8")
    print("Wrote", path.resolve())


if __name__ == "__main__":
    main()

