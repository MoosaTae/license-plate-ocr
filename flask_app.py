import base64
import io
import os

import easyocr
import numpy as np
from flask import Flask, abort, render_template_string, request
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

app = Flask(__name__)

reader = easyocr.Reader(["th"], gpu=False)
ALLOWLIST = "".join(chr(c) for c in range(0x0E01, 0x0E3B)) + "0123456789"

CONFIDENCE_THRESHOLD = 0.2


def load_thai_provinces():
    """Load list of Thai provinces from file."""
    try:
        with open("province_list.txt", "r", encoding="utf-8") as f:
            provinces = [line.strip() for line in f.readlines() if line.strip()]
        return provinces
    except FileNotFoundError:
        print("Warning: province_list.txt not found")
        return []


THAI_PROVINCES = load_thai_provinces()
print(f"Loaded {len(THAI_PROVINCES)} Thai provinces for validation")
try:
    FONT_EN = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 22)
except:
    try:
        FONT_EN = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22
        )
    except:
        FONT_EN = ImageFont.load_default()

FONT_TH = FONT_EN


def pil_to_base64(img, quality=90):
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def aggressive_preprocess(img):
    """Aggressive preprocessing for better OCR accuracy (from Jupyter experiments)."""
    # Higher contrast enhancement (1.8x vs 1.25x)
    img_contrast = ImageEnhance.Contrast(img).enhance(1.8)

    # More aggressive sharpening (200% vs 120%)
    img_sharp = img_contrast.filter(
        ImageFilter.UnsharpMask(radius=2.0, percent=200, threshold=1)
    )

    return img_sharp


def standard_preprocess(img):
    """Standard preprocessing (original approach)."""
    img2 = ImageEnhance.Contrast(img).enhance(1.25)
    img2 = img2.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return img2


def run_improved_ocr(img):
    """Run OCR with best parameters discovered from Jupyter experiments."""
    # Use aggressive preprocessing
    processed_img = aggressive_preprocess(img)
    arr = np.array(processed_img)

    # Optimized parameters from experiments
    results = reader.readtext(
        arr,
        detail=1,
        paragraph=False,
        contrast_ths=0.02,  # More sensitive (vs 0.05)
        adjust_contrast=0.9,  # Stronger adjustment (vs 0.7)
        text_threshold=0.4,  # Lower threshold (vs 0.6)
        low_text=0.15,  # Detect fainter text (vs 0.3)
        link_threshold=0.3,  # Better linking (vs 0.4)
        allowlist=ALLOWLIST,
        decoder="beamsearch",
        rotation_info=[0, 90, -90],
    )

    return results, processed_img


def run_standard_ocr(img):
    """Run OCR with standard parameters."""
    processed_img = standard_preprocess(img)
    arr = np.array(processed_img)
    results = reader.readtext(
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
    return results, processed_img


def filter_relevant_detections(results):
    """Filter out irrelevant detections like long numbers and noise."""
    import re

    filtered = []
    for bbox, text, conf in results:
        text_clean = re.sub(r"\s+", " ", text.strip())

        # Skip very long numbers (likely metadata/background)
        if re.match(r"^\d{8,}$", text_clean):
            continue

        # Skip very short single characters with low confidence
        if len(text_clean) == 1 and conf < 0.7:
            continue

        # Skip empty or whitespace-only text
        if not text_clean:
            continue

        filtered.append((bbox, text_clean, conf))

    return filtered


def run_ocr(img):
    """Main OCR function using best practices from experiments."""
    # Try improved OCR first
    results_improved, processed_img = run_improved_ocr(img)
    filtered_improved = filter_relevant_detections(results_improved)

    # If improved OCR finds good results, use them
    high_conf_improved = [
        (bbox, text, conf) for bbox, text, conf in filtered_improved if conf >= 0.3
    ]

    if high_conf_improved:
        print(
            f"Using IMPROVED OCR: Found {len(high_conf_improved)} high-confidence detections"
        )
        return filtered_improved, processed_img, "improved"

    # Fallback to standard OCR if improved doesn't find much
    results_standard, processed_std = run_standard_ocr(img)
    filtered_standard = filter_relevant_detections(results_standard)

    # Return the approach that found more high-confidence detections
    high_conf_standard = [
        (bbox, text, conf) for bbox, text, conf in filtered_standard if conf >= 0.3
    ]

    if len(high_conf_improved) >= len(high_conf_standard):
        print(
            f"Using IMPROVED OCR: {len(high_conf_improved)} vs {len(high_conf_standard)} detections"
        )
        return filtered_improved, processed_img, "improved"
    else:
        print(
            f"Using STANDARD OCR: {len(high_conf_standard)} vs {len(high_conf_improved)} detections"
        )
        return filtered_standard, processed_std, "standard"


def similarity_score(text1, text2):
    """Calculate similarity between two strings."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, text1, text2).ratio()


def is_valid_province(detected_text, threshold=0.8):
    """Check if detected text matches any Thai province."""
    detected_clean = detected_text.strip()

    # First pass: Check for exact matches
    for province in THAI_PROVINCES:
        if detected_clean == province:
            return True, f"Exact province match: {province}"

    # Second pass: Check for partial matches
    for province in THAI_PROVINCES:
        if detected_clean in province or province in detected_clean:
            return True, f"Partial province match: {province}"

    # Third pass: Check for fuzzy matches (only for very close matches)
    best_match = None
    best_score = 0
    for province in THAI_PROVINCES:
        similarity = similarity_score(detected_clean, province)
        if similarity >= threshold and similarity > best_score:
            best_score = similarity
            best_match = province

    if best_match:
        return True, f"Fuzzy province match: {best_match} ({best_score:.2f})"

    return False, "Not a recognized Thai province"


def extract_license_components(detected_text):
    """Extract license plate components (letters and numbers)."""
    import re

    # Thai license plate patterns
    # Format: XX 1234, XX-1234, XXXX 1234, etc.

    # Remove extra spaces and normalize
    cleaned = re.sub(r"\s+", " ", detected_text.strip())

    # Extract Thai letters
    thai_letters = re.findall(r"[ก-๙]+", cleaned)

    # Extract numbers
    numbers = re.findall(r"\d+", cleaned)

    return {
        "full_text": cleaned,
        "thai_letters": thai_letters,
        "numbers": numbers,
        "has_thai": len(thai_letters) > 0,
        "has_numbers": len(numbers) > 0,
    }


def validate_license_plate(detected_text, confidence, threshold=CONFIDENCE_THRESHOLD):
    """Enhanced validation with confidence + province checking."""
    # Step 1: Check confidence threshold
    if confidence < threshold:
        return False, f"Low confidence: {confidence:.3f} < {threshold}"

    # Step 2: Extract license plate components
    components = extract_license_components(detected_text)

    # Step 3: Check if it looks like a license plate
    if not components["has_thai"] and not components["has_numbers"]:
        return False, f"Invalid format: No Thai letters or numbers found"

    # Step 4: If it contains Thai text, check against provinces
    if components["thai_letters"]:
        for thai_part in components["thai_letters"]:
            is_province, province_reason = is_valid_province(thai_part)
            if is_province:
                return True, f"Valid license plate: {province_reason}"

    # Step 5: If no province match but has valid format
    if components["has_thai"] and components["has_numbers"]:
        return (
            True,
            f"Valid format: Thai letters + numbers (confidence: {confidence:.3f})",
        )

    # Step 6: Numbers only (might be license number part)
    if components["has_numbers"] and not components["has_thai"]:
        return (
            True,
            f"License number: {components['numbers']} (confidence: {confidence:.3f})",
        )

    # Step 7: Default to confidence-based validation
    return True, f"High confidence: {confidence:.3f} >= {threshold}"


def draw_mixed_text(draw, pos, text, fill=(255, 255, 255)):
    """Draw text using appropriate font."""
    x, y = pos
    for ch in text:
        font = FONT_TH if "\u0e00" <= ch <= "\u0e7f" else FONT_EN
        draw.text((x, y), ch, font=font, fill=fill)
        try:
            x += font.getlength(ch)
        except:
            x += 10  # Fallback spacing


def draw_annotation(img, results):
    """Draw bounding boxes and annotations on image."""
    draw = ImageDraw.Draw(img)

    for bbox, text, conf in results:
        # Validate the detected text
        is_valid, reason = validate_license_plate(text, conf)

        # Choose colors based on validation result
        if is_valid:
            color = (0, 255, 0)  # Green for PASS
            status = "PASS"
        else:
            color = (255, 0, 0)  # Red for FAIL
            status = "FAIL"

        # Draw bounding box
        pts = [(int(x), int(y)) for x, y in bbox]
        draw.polygon(pts, outline=color, width=3)

        # Draw text label
        label = f"{text} ({conf:.2f}) - {status}"
        text_position = (pts[0][0], max(pts[0][1] - 30, 0))
        draw_mixed_text(draw, text_position, label, fill=color)


# HTML template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>License Plate OCR</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .upload-form { text-align: center; margin: 30px 0; padding: 20px; border: 2px dashed #ccc; border-radius: 10px; }
        .upload-form input[type="file"] { margin: 10px; }
        .upload-form input[type="submit"] { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .upload-form input[type="submit"]:hover { background-color: #0056b3; }
        .results { margin-top: 30px; }
        .image-result { text-align: center; margin: 20px 0; }
        .image-result img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .text-results { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .text-results { white-space: pre-wrap; font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; }
        .text-results h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        .text-results h3 { color: #555; margin-top: 20px; }
        .text-results code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; font-family: 'Courier New', monospace; }
        .text-results strong { color: #333; }
        .text-results hr { border: 1px solid #ddd; margin: 20px 0; }
        .pass { color: #28a745; }
        .fail { color: #dc3545; }
    </style>
    <script>
        function parseMarkdown(text) {
            return text
                .replace(/^## (.*$)/gim, '<h2>$1</h2>')
                .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/^- (.*$)/gim, '<li>$1</li>')
                .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
                .replace(/^---$/gim, '<hr>')
                .replace(/PASS/g, '<span style="color: #28a745;">PASS</span>')
                .replace(/FAIL/g, '<span style="color: #dc3545;">FAIL</span>')
                .replace(/\n/g, '<br>');
        }

        window.onload = function() {
            const markdownContent = document.getElementById('markdown-content');
            if (markdownContent) {
                const rawText = markdownContent.textContent;
                markdownContent.innerHTML = parseMarkdown(rawText);
            }
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>License Plate OCR System</h1>

        <div>
            <h3>Advanced OCR System (Jupyter-Optimized)</h3>
            <p><strong>Confidence Threshold:</strong> {{ confidence_threshold }}</p>
            <p><strong>Province Database:</strong> 77 Thai provinces loaded</p>
            <p><strong>Enhanced Features:</strong></p>
            <ul>
                <li>Aggressive preprocessing (1.8x contrast, 200% sharpening)</li>
                <li>Optimized OCR parameters (contrast_ths=0.02, text_threshold=0.4)</li>
                <li>Noise filtering (removes metadata, background text)</li>
                <li>Thai province recognition with fuzzy matching</li>
                <li>License plate format validation</li>
            </ul>
        </div>

        <form class="upload-form" method="post" enctype="multipart/form-data">
            <h3>Upload License Plate Image</h3>
            <input type="file" name="image" accept="image/*" required>
            <br>
            <input type="submit" value="Analyze License Plate">
        </form>

        {% if result_b64 %}
        <div class="results">
            <div class="image-result">
                <h3>OCR Results</h3>
                <img src="data:image/jpeg;base64,{{ result_b64 }}" alt="OCR Result">
            </div>

            <div class="text-results">
                <h3>Analysis Results</h3>
                <div id="markdown-content">{{ raw_text | safe }}</div>
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("image")
        if not f:
            return abort(400, "No file uploaded")

        try:
            img = Image.open(f.stream).convert("RGB")
            results, processed_img, ocr_method = run_ocr(img)

            # Format results text in markdown
            if results:
                raw_text_lines = []
                raw_text_lines.append(f"## OCR Analysis Report")
                raw_text_lines.append(f"**Method Used:** `{ocr_method.upper()}`")
                raw_text_lines.append(f"**Total Detections:** {len(results)}")
                raw_text_lines.append("")

                passed_count = 0
                for i, (bbox, text, conf) in enumerate(results, 1):
                    is_valid, reason = validate_license_plate(text, conf)
                    status = "PASS" if is_valid else "FAIL"
                    if is_valid:
                        passed_count += 1

                    status_icon = "PASS" if is_valid else "FAIL"
                    raw_text_lines.append(f"### Detection #{i}")
                    raw_text_lines.append(f"- **Text:** `{text}`")
                    raw_text_lines.append(f"- **Confidence:** `{conf:.3f}`")
                    raw_text_lines.append(f"- **Status:** {status_icon} **{status}**")
                    raw_text_lines.append(f"- **Reason:** {reason}")
                    raw_text_lines.append("")

                raw_text_lines.append("---")
                raw_text_lines.append(f"### Summary")
                raw_text_lines.append(
                    f"- **Passed Validation:** {passed_count}/{len(results)}"
                )
                raw_text_lines.append(
                    f"- **Success Rate:** {(passed_count/len(results)*100):.1f}%"
                )

                raw_text = "\n".join(raw_text_lines)
            else:
                raw_text = f"## OCR Analysis Report\n\n**Method Used:** `{ocr_method.upper()}`\n\n**No text detected**"

            # Draw annotations
            draw_annotation(processed_img, results)

            return render_template_string(
                HTML,
                result_b64=pil_to_base64(processed_img),
                raw_text=raw_text,
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )
        except Exception as e:
            return abort(500, f"Error processing image: {str(e)}")

    return render_template_string(
        HTML,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )


if __name__ == "__main__":
    print("Starting License Plate OCR Flask App...")
    app.run(host="0.0.0.0", port=5000, debug=True)
