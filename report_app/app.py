
import os
import re
import json
import uuid
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from google import genai
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
REPORT_DIR = APP_DIR / "generated_reports"

RESULT_DIR = ROOT / "results" / "avater_meld_v3"
INFERENCE_SCRIPT = ROOT / "fusion" / "avater_meld_v3" / "inference.py"

TEST_RESULTS = RESULT_DIR / "test_results.json"
TRAINING_HISTORY = RESULT_DIR / "training_history.json"
CONFUSION_MATRIX = RESULT_DIR / "confusion_matrix.csv"
TEST_PREDICTIONS = RESULT_DIR / "test_predictions.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(APP_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

MAX_UPLOAD_BYTES = 250 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

LABELS = [
    "neutral", "joy", "sadness", "anger",
    "fear", "disgust", "surprise"
]

app = FastAPI(
    title="AVaTER-MELD V3 Research Report Generator",
    version="2.0.0"
)


# =========================================================
# EVALUATION EVIDENCE
# Read-only access to existing V3 experiment files
# =========================================================

def load_evaluation_evidence():
    required = [
        TEST_RESULTS,
        TRAINING_HISTORY,
        CONFUSION_MATRIX,
        TEST_PREDICTIONS
    ]

    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(
            "Required V3 evaluation files are missing: "
            + ", ".join(missing)
        )

    with TEST_RESULTS.open("r", encoding="utf-8") as f:
        results = json.load(f)

    with TRAINING_HISTORY.open("r", encoding="utf-8") as f:
        history = json.load(f)

    confusion_df = pd.read_csv(CONFUSION_MATRIX)
    predictions_df = pd.read_csv(TEST_PREDICTIONS)

    if not isinstance(history, list):
        raise ValueError("Unexpected training history format.")

    if not {"clip_id", "true_label", "predicted_label"}.issubset(
        predictions_df.columns
    ):
        raise ValueError("Unexpected test_predictions.csv schema.")

    # Calculate the main confusion patterns from actual test predictions.
    confusion_counts = {}
    grouped = predictions_df.groupby(
        ["true_label", "predicted_label"]
    ).size()

    for (actual, predicted), count in grouped.items():
        if str(actual) != str(predicted):
            key = f"{actual} -> {predicted}"
            confusion_counts[key] = int(count)

    top_confusions = sorted(
        confusion_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )[:15]

    # Keep the prompt compact while preserving useful training trends.
    history_summary = []
    for item in history:
        train = item.get("train", {})
        dev = item.get("dev", {})
        history_summary.append({
            "epoch": item.get("epoch"),
            "train_loss": train.get("loss"),
            "train_accuracy": train.get("accuracy"),
            "train_macro_f1": train.get("macro_f1"),
            "dev_loss": dev.get("loss"),
            "dev_accuracy": dev.get("accuracy"),
            "dev_macro_f1": dev.get("macro_f1"),
            "learning_rate": item.get("learning_rate")
        })

    return {
        "experiment": results.get("experiment"),
        "description": results.get("description"),
        "best_epoch": results.get("best_epoch"),
        "best_dev_macro_f1": results.get("best_dev_macro_f1"),
        "test_loss": results.get("test_loss"),
        "test_accuracy": results.get("test_accuracy"),
        "test_macro_f1": results.get("test_macro_f1"),
        "test_weighted_f1": results.get("test_weighted_f1"),
        "labels": results.get("labels", LABELS),
        "train_samples": results.get("train_samples"),
        "dev_samples": results.get("dev_samples"),
        "test_samples": results.get("test_samples"),
        "classification_report": results.get("classification_report", {}),
        "saved_confusion_matrix": results.get("confusion_matrix", []),
        "confusion_matrix_csv_columns": confusion_df.columns.tolist(),
        "top_test_confusions": [
            {"pattern": key, "count": count}
            for key, count in top_confusions
        ],
        "training_history": history_summary,
        "test_prediction_rows": int(len(predictions_df))
    }


# =========================================================
# EXISTING V3 INFERENCE
# Does not retrain or edit the checkpoint
# =========================================================

def parse_inference_output(output):
    prediction_match = re.search(
        r"Predicted emotion:\s*(\w+)", output, re.IGNORECASE
    )
    score_match = re.search(
        r"Model softmax score:\s*([0-9]*\.?[0-9]+)", output
    )
    transcript_match = re.search(
        r"Transcript:\s*(.*)", output
    )

    if not prediction_match:
        raise ValueError(
            "V3 output did not contain a predicted emotion."
        )

    scores = {}
    in_scores = False

    for line in output.splitlines():
        if line.strip() == "Emotion scores:":
            in_scores = True
            continue

        if in_scores:
            match = re.match(
                r"^\s*(neutral|joy|sadness|anger|fear|disgust|surprise)"
                r"\s*:\s*([0-9]*\.?[0-9]+)\s*$",
                line,
                re.IGNORECASE
            )
            if match:
                scores[match.group(1).lower()] = float(match.group(2))

    if set(scores) != set(LABELS):
        raise ValueError(
            "V3 output did not contain all seven emotion scores."
        )

    transcript = (
        transcript_match.group(1).strip()
        if transcript_match else ""
    )
    if transcript == "(no speech recognized)":
        transcript = ""

    return {
        "predicted_emotion": prediction_match.group(1).lower(),
        "model_softmax_score": (
            float(score_match.group(1)) if score_match else None
        ),
        "emotion_scores": scores,
        "transcript": transcript
    }


def run_v3_inference(video_path):
    if not INFERENCE_SCRIPT.is_file():
        raise RuntimeError("Existing V3 inference script was not found.")

    command = [
        os.sys.executable,
        str(INFERENCE_SCRIPT),
        "--video",
        str(video_path)
    ]

    result = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
        errors="replace"
    )

    if result.returncode != 0:
        detail = result.stderr or result.stdout or "Unknown inference error"
        raise RuntimeError("V3 inference failed: " + detail[-2500:])

    return parse_inference_output(result.stdout)


# =========================================================
# GEMINI ANALYSIS
# Uses measured evaluation evidence + this video's output
# =========================================================

def generate_llm_report(video_evidence, evaluation):
    if (
        not GEMINI_API_KEY
        or "PASTE_YOUR" in GEMINI_API_KEY
        or "YOUR_ACTUAL" in GEMINI_API_KEY
    ):
        raise RuntimeError(
            "Gemini API key is not configured in report_app/.env."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    evidence = {
        "new_video_inference": video_evidence,
        "saved_v3_evaluation": evaluation
    }

    prompt = f"""
You are an analytical research assistant writing a rigorous report
for the AVaTER-MELD V3 multimodal emotion-recognition research project.

Analyze the supplied evidence and produce a professional research report.

EVIDENCE JSON:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Write these sections:

1. Executive Summary
2. System and Experimental Setup
3. Saved V3 Test Performance
4. Class-Wise Performance Analysis
5. Confusion Patterns and Error Analysis
6. Training and Validation Trends
7. Analysis of the Newly Submitted Video
8. Relationship Between This Prediction and Overall Model Performance
9. Limitations and Reliability
10. Recommendations for Future Work
11. Conclusion

STRICT EVIDENCE RULES:
- Use the supplied numeric results accurately.
- Convert accuracy, F1 and other proportion metrics to percentages when useful.
- Explain that test accuracy measures the fraction of test predictions
  classified correctly; macro-F1 weights classes equally, whereas
  weighted-F1 accounts for class support.
- Discuss class-wise precision, recall, F1 and support when available.
- Identify the strongest and weakest class performance using the actual
  supplied values.
- Use confusion patterns from the actual test predictions.
- Discuss training behaviour using the supplied training history.
  Do not claim overfitting or underfitting without supporting trends.
- Keep the saved test-set evaluation separate from the new video's output.
- The individual video's softmax score is a model output, not a calibrated
  probability that the real-world emotion is correct.
- Do not claim that the new video has a known true label.
- The supplied video data includes a transcript and seven class scores,
  but no detailed per-frame face descriptions or acoustic explanations.
  Do not invent facial expressions, vocal tone, identity, context,
  diagnoses or hidden mental states.
- Suggest possible future experiments, clearly labelled as recommendations,
  not completed results.
- Never fabricate metrics, sources, comparisons, or experimental outcomes.
- Gemini interprets the evidence; the V3 model performs classification.
- Use plain text with clear numbered section headings.
- Avoid Markdown tables, code fences and unsupported claims.
- If the transcript is empty, state that no speech was recognized.

Write enough detail for an academic project report, with meaningful
interpretation rather than merely restating the numbers.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty report.")

    return text


# =========================================================
# PDF REPORT
# =========================================================

def create_pdf(video_evidence, evaluation, report_text, pdf_path):
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#253575"),
        spaceAfter=14
    ))

    styles.add(ParagraphStyle(
        name="SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#253575"),
        spaceBefore=12,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        spaceAfter=7
    ))

    styles.add(ParagraphStyle(
        name="SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569")
    ))

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.7 * cm,
        title="AVaTER-MELD V3 Research Analysis Report",
        author="AVaTER-MELD V3 Gemini Report Generator"
    )

    story = [
        Paragraph("AVaTER-MELD V3", styles["ReportTitleCustom"]),
        Paragraph(
            "Multimodal Emotion Recognition: Research Analysis Report",
            styles["Heading1"]
        ),
        Spacer(1, 8),
        Paragraph("1. Video Inference Summary", styles["SectionCustom"])
    ]

    summary_rows = [
        ["Field", "Value"],
        ["Predicted emotion", video_evidence["predicted_emotion"].title()],
        [
            "Model softmax score",
            (
                f'{video_evidence["model_softmax_score"]:.4f}'
                if video_evidence["model_softmax_score"] is not None
                else "Not available"
            )
        ],
        ["Evaluation experiment", str(evaluation.get("experiment", "N/A"))],
        [
            "Saved test accuracy",
            f'{evaluation["test_accuracy"] * 100:.2f}%'
        ],
        [
            "Saved test macro-F1",
            f'{evaluation["test_macro_f1"] * 100:.2f}%'
        ],
        [
            "Saved test weighted-F1",
            f'{evaluation["test_weighted_f1"] * 100:.2f}%'
        ],
        ["Saved test sample count", str(evaluation.get("test_samples", "N/A"))],
        ["Best development epoch", str(evaluation.get("best_epoch", "N/A"))]
    ]

    table = Table(summary_rows, colWidths=[6 * cm, 9.5 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#253575")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#E8ECFA")),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
    ]))
    story.extend([table, Spacer(1, 10)])

    story.append(Paragraph("2. All Seven Video Emotion Scores", styles["SectionCustom"]))
    score_rows = [["Emotion", "V3 model score"]]
    for label in LABELS:
        score_rows.append([
            label.title(),
            f'{video_evidence["emotion_scores"][label]:.4f}'
        ])

    score_table = Table(score_rows, colWidths=[7 * cm, 8.5 * cm], repeatRows=1)
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#253575")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#F4F6FA")
        ]),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
    ]))
    story.extend([score_table, Spacer(1, 10)])

    story.append(Paragraph("3. Recognized Speech Transcript", styles["SectionCustom"]))
    transcript = video_evidence.get("transcript") or "(No speech recognized.)"
    story.append(Paragraph(escape(transcript), styles["BodyCustom"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("4. Gemini Research Analysis", styles["SectionCustom"]))

    for line in report_text.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue

        safe_line = escape(line)

        if re.match(r"^\d+[\.\)]\s+", line) or (
            len(line) < 110 and line.endswith(":")
        ):
            story.append(Paragraph(safe_line, styles["SectionCustom"]))
        else:
            story.append(Paragraph(safe_line, styles["BodyCustom"]))

    story.extend([
        Spacer(1, 12),
        Paragraph(
            "Research limitation: results summarize an experimental model "
            "and its saved MELD test evaluation. A single prediction does "
            "not establish a person's actual emotional state. Model softmax "
            "scores are not calibrated probabilities of real-world correctness. "
            "Gemini-generated interpretations are analytical commentary, "
            "not new experimental measurements.",
            styles["SmallCustom"]
        )
    ])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(
            1.7 * cm, 0.9 * cm,
            "AVaTER-MELD V3 | Gemini-assisted research report"
        )
        canvas.drawRightString(
            A4[0] - 1.7 * cm, 0.9 * cm,
            f"Page {doc.page}"
        )
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)


# =========================================================
# WEB INTERFACE
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>AVaTER-MELD V3 Research Reports</title>
      <style>
        * { box-sizing: border-box; }
        body {
          margin: 0; min-height: 100vh; padding: 30px 16px;
          display: flex; justify-content: center; align-items: center;
          font-family: Arial, sans-serif; background: #f1f5f9; color: #172033;
        }
        main {
          max-width: 680px; width: 100%; background: white;
          padding: 32px; border-radius: 18px;
          box-shadow: 0 12px 40px #0f172a12;
        }
        h1 { margin-top: 10px; }
        p { color: #536176; line-height: 1.6; }
        .tag {
          display: inline-block; background: #e0e7ff; color: #3730a3;
          padding: 7px 11px; border-radius: 20px; font-size: 12px;
          font-weight: bold;
        }
        .upload {
          border: 2px dashed #a5b4fc; background: #f8faff;
          padding: 24px; border-radius: 12px; margin: 22px 0;
        }
        input { width: 100%; margin-top: 12px; }
        button {
          width: 100%; padding: 14px; border: 0; border-radius: 9px;
          background: #3730a3; color: white; font-size: 16px;
          font-weight: bold; cursor: pointer;
        }
        button:disabled { opacity: .6; cursor: wait; }
        #status { white-space: pre-wrap; }
        .note { font-size: 13px; }
      </style>
    </head>
    <body>
      <main>
        <span class="tag">V3 + GEMINI ANALYTICS</span>
        <h1>Emotion Research Report</h1>
        <p>
          Run your existing AVaTER-MELD V3 inference pipeline.
          Gemini analyzes the new video's output alongside the saved
          V3 test metrics, class-wise evaluation, confusion patterns
          and training history, then prepares a PDF report.
        </p>
        <form id="uploadForm">
          <div class="upload">
            <strong>Select a video for inference</strong>
            <p class="note">MP4, MOV, AVI, MKV or WEBM · Maximum 250 MB</p>
            <input type="file" name="video"
              accept=".mp4,.mov,.avi,.mkv,.webm" required>
          </div>
          <button id="submitButton" type="submit">
            Run V3 + Generate Research PDF
          </button>
        </form>
        <p id="status" role="status"></p>
        <p class="note">
          Existing model and evaluation files are read only.
          Model scores are not proof of a person's actual emotional state.
        </p>
      </main>
      <script>
        const form = document.getElementById("uploadForm");
        const button = document.getElementById("submitButton");
        const statusBox = document.getElementById("status");

        form.addEventListener("submit", async (event) => {
          event.preventDefault();
          const fileInput = form.querySelector('input[name="video"]');
          if (!fileInput.files.length) return;

          button.disabled = true;
          statusBox.textContent =
            "Running V3 inference, loading evaluation results, and asking Gemini to analyze the evidence. Please wait.";

          try {
            const response = await fetch("/generate-report", {
              method: "POST",
              body: new FormData(form)
            });

            if (!response.ok) {
              const error = await response.json().catch(() => ({}));
              throw new Error(error.detail || "Report generation failed.");
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "AVaTER_MELD_V3_Research_Report.pdf";
            document.body.appendChild(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);

            statusBox.textContent =
              "Research PDF generated successfully. Download started.";
          } catch (error) {
            statusBox.textContent = "Error: " + error.message;
          } finally {
            button.disabled = false;
          }
        });
      </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "ok",
        "inference_script_exists": INFERENCE_SCRIPT.is_file(),
        "test_results_exists": TEST_RESULTS.is_file(),
        "training_history_exists": TRAINING_HISTORY.is_file(),
        "confusion_matrix_exists": CONFUSION_MATRIX.is_file(),
        "test_predictions_exists": TEST_PREDICTIONS.is_file(),
        "gemini_key_configured": bool(
            GEMINI_API_KEY
            and "PASTE_YOUR" not in GEMINI_API_KEY
            and "YOUR_ACTUAL" not in GEMINI_API_KEY
        )
    }


@app.post("/generate-report")
async def generate_report(video: UploadFile = File(...)):
    original_name = Path(video.filename or "video.mp4").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    upload_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{upload_id}{extension}"
    pdf_path = REPORT_DIR / f"research_report_{upload_id}.pdf"

    try:
        size = 0

        with video_path.open("wb") as output:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break

                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 250 MB upload limit."
                    )

                output.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty."
            )

        # Read actual saved experimental results before processing.
        evaluation = load_evaluation_evidence()

        # Run the unchanged V3 inference script on the new video.
        video_evidence = run_v3_inference(video_path)
        video_evidence["filename"] = original_name

        # Gemini interprets both the measured metrics and video output.
        report_text = generate_llm_report(video_evidence, evaluation)

        # Save only the newly generated PDF in the report_app folder.
        create_pdf(video_evidence, evaluation, report_text, pdf_path)

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="AVaTER_MELD_V3_Research_Report.pdf"
        )

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="V3 inference timed out. Try a shorter video."
        )
    except Exception as exc:
        print(f"Report generation error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Report generation failed: {type(exc).__name__}. "
                "Check the server terminal for details."
            )
        )
    finally:
        await video.close()
        video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
