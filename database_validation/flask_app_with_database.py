import base64
import io
import os

import easyocr
import numpy as np
from flask import Flask, abort, render_template_string, request
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from license_plate_validator import LicensePlateDatabase

app = Flask(__name__)

reader = easyocr.Reader(["th"], gpu=False)
ALLOWLIST = "".join(chr(c) for c in range(0x0E01, 0x0E3B)) + "0123456789"

CONFIDENCE_THRESHOLD = 0.2

# Initialize database
license_db = LicensePlateDatabase("license_plate_database.csv")


def aggressive_preprocess(img):
    """More aggressive preprocessing for difficult images."""
    img_contrast = ImageEnhance.Contrast(img).enhance(1.8)
    img_sharp = img_contrast.filter(
        ImageFilter.UnsharpMask(radius=2.0, percent=200, threshold=1)
    )
    return img_sharp


def run_improved_ocr(img):
    """Run OCR with improved parameters for better accuracy."""
    processed_img = aggressive_preprocess(img)
    arr = np.array(processed_img)

    results = reader.readtext(
        arr,
        detail=1,
        paragraph=False,
        contrast_ths=0.02,
        adjust_contrast=0.9,
        text_threshold=0.4,
        low_text=0.15,
        link_threshold=0.3,
        allowlist=ALLOWLIST,
        decoder="beamsearch",
        rotation_info=[0, 90, -90],
    )
    return results, processed_img


def filter_relevant_detections(results):
    """Filter out irrelevant detections like long numbers and noise."""
    import re

    filtered = []
    for bbox, text, conf in results:
        text_clean = re.sub(r"\s+", " ", text.strip())

        if re.match(r"^\d{8,}$", text_clean):
            continue

        if len(text_clean) == 1 and conf < 0.7:
            continue

        if not text_clean:
            continue

        filtered.append((bbox, text_clean, conf))

    return filtered


def run_ocr(img):
    """Main OCR function using best practices from experiments."""
    results_improved, processed_img = run_improved_ocr(img)
    filtered_improved = filter_relevant_detections(results_improved)

    high_conf_improved = [
        (bbox, text, conf) for bbox, text, conf in filtered_improved if conf >= 0.3
    ]

    if len(high_conf_improved) >= 2:
        return high_conf_improved, processed_img

    arr = np.array(img)
    results_standard = reader.readtext(
        arr,
        detail=1,
        paragraph=False,
        contrast_ths=0.05,
        adjust_contrast=0.7,
        text_threshold=0.6,
        low_text=0.3,
        link_threshold=0.4,
        allowlist=ALLOWLIST,
        decoder="beamsearch",
        rotation_info=[0, 90, -90],
    )

    filtered_standard = filter_relevant_detections(results_standard)
    return filtered_standard, img


def draw_results(img, results):
    """Draw OCR results on image."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    try:
        font = ImageFont.truetype("Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()

    for i, (bbox, text, conf) in enumerate(results):
        points = np.array(bbox).astype(int)
        draw.polygon([tuple(p) for p in points], outline="red", width=2)

        text_position = (points[0][0], max(0, points[0][1] - 25))
        draw.text(text_position, f"{i+1}: {text} ({conf:.2f})", fill="red", font=font)

    return img_copy


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>License Plate OCR with Database Validation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-area:hover { border-color: #999; }
        .result-container { margin: 20px 0; }
        .detection-item { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .pass { border-left: 5px solid #4CAF50; }
        .fail { border-left: 5px solid #f44336; }
        .database-match { background: #e8f5e9; padding: 10px; margin: 10px 0; border-radius: 3px; }
        img { max-width: 100%; height: auto; }
        .stats { background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🚗 License Plate OCR with Database Validation</h1>

    <form method="post" enctype="multipart/form-data">
        <div class="upload-area">
            <input type="file" name="image" accept="image/*" required>
            <p>Select an image of a Thai license plate</p>
        </div>
        <button type="submit">Analyze License Plate</button>
    </form>

    {% if results %}
    <div class="result-container">
        <h2>📊 Analysis Results</h2>

        {% if original_image %}
        <h3>Original Image</h3>
        <img src="data:image/png;base64,{{ original_image }}" alt="Original">
        {% endif %}

        {% if processed_image %}
        <h3>Processed Image with Detections</h3>
        <img src="data:image/png;base64,{{ processed_image }}" alt="Processed">
        {% endif %}

        <h3>Database Validation Results</h3>
        {% for i, result in enumerate(results) %}
        <div class="detection-item {{ 'pass' if result.validation_status == 'PASS' else 'fail' }}">
            <h4>Detection #{{ i+1 }}</h4>
            <div id="detection-{{ i+1 }}"></div>

            {% if result.database_match %}
            <div class="database-match">
                <strong>📋 Database Record:</strong><br>
                <strong>Plate:</strong> {{ result.database_match.plate_number }}<br>
                <strong>Province:</strong> {{ result.database_match.province }}<br>
                <strong>Vehicle Type:</strong> {{ result.database_match.vehicle_type }}<br>
                <strong>Status:</strong> {{ result.database_match.status }}
            </div>
            {% endif %}
        </div>
        {% endfor %}

        <div class="stats">
            <h3>📈 Database Statistics</h3>
            <div id="stats"></div>
        </div>
    </div>

    <script>
        const results = {{ results | tojsonfilter }};
        const stats = {{ stats | tojsonfilter }};

        results.forEach((result, index) => {
            const detectionDiv = document.getElementById(`detection-${index + 1}`);
            const markdown = `
**Text:** \`${result.detected_plate}\`
**Confidence:** \`${result.confidence.toFixed(3)}\`
**Status:** ${result.validation_status === 'PASS' ? '✅ **PASS**' : '❌ **FAIL**'}
**Reason:** ${result.validation_reason}
${result.match_type ? `**Match Type:** ${result.match_type} (${result.match_score.toFixed(2)})` : ''}
            `.trim();

            detectionDiv.innerHTML = markdown
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\`(.*?)\`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        });

        const statsDiv = document.getElementById('stats');
        const statsMarkdown = `
**Total Plates in Database:** ${stats.total}
**Provinces:** ${Object.keys(stats.by_province).join(', ')}
**Vehicle Types:** ${Object.keys(stats.by_vehicle_type).join(', ')}
**Status Distribution:** ${Object.entries(stats.by_status).map(([k,v]) => `${k} (${v})`).join(', ')}
        `.trim();

        statsDiv.innerHTML = statsMarkdown
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
    </script>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        stats = license_db.get_statistics()
        return render_template_string(HTML_TEMPLATE, stats=stats)

    file = request.files.get("image")
    if not file:
        abort(400)

    try:
        img = Image.open(file.stream).convert("RGB")
        results, processed_img = run_ocr(img)

        validation_results = []
        for bbox, text, conf in results:
            validation = license_db.validate_license_plate(
                text, conf, CONFIDENCE_THRESHOLD
            )
            validation_results.append(validation)

        img_with_boxes = draw_results(processed_img, results)

        original_buffer = io.BytesIO()
        img.save(original_buffer, format="PNG")
        original_b64 = base64.b64encode(original_buffer.getvalue()).decode()

        processed_buffer = io.BytesIO()
        img_with_boxes.save(processed_buffer, format="PNG")
        processed_b64 = base64.b64encode(processed_buffer.getvalue()).decode()

        stats = license_db.get_statistics()

        return render_template_string(
            HTML_TEMPLATE,
            results=validation_results,
            original_image=original_b64,
            processed_image=processed_b64,
            stats=stats
        )

    except Exception as e:
        return f"Error processing image: {str(e)}", 500


if __name__ == "__main__":
    print(f"License Plate Database loaded with {len(license_db.database)} records")
    print("Starting Flask app...")
    app.run(debug=True, port=5000)