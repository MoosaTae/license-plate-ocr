# License Plate OCR Lab

This project implements a complete license plate OCR system using EasyOCR for Thai license plates.
![alt text](img/app.png)
## Features

- **EasyOCR Integration**: Uses EasyOCR with Thai character support
- **Image Preprocessing**: Contrast enhancement and unsharp masking for better OCR accuracy
- **Confidence Filtering**: Configurable confidence threshold to filter low-quality detections

- **Visualization**: Draws bounding boxes with Pass/Fail status on detected text
- **Web Interface**: Flask-based web application for easy testing
- **Jupyter Notebook**: Complete experimental notebook with parameter tuning

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv --system-site-packages env-license-plate-ocr
   source env-license-plate-ocr/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install flask gdown easyocr pillow numpy --no-cache-dir
   ```

3. **Download sample images:**
   ```bash
   gdown 1rGHs_bV7CH34lzvvFgCW6LNNKTGjxufj -O license-plate-th.jpg
   gdown 1sZ0h6eRmFaYFKX9SXGuMBKV_0Xrhehf6 -O license-plate-th2.jpg
   gdown 1o-EobL07p_EU7PCo0100AJ_cLR_2mJDH -O license-plate-th3.jpg
   ```

## Usage

### Jupyter Notebook (Recommended for Experiments)

1. **Start Jupyter:**
   ```bash
   source env-license-plate-ocr/bin/activate
   jupyter notebook license_plate_ocr_lab.ipynb
   ```

2. **Run all cells** to see:
   - Basic OCR implementation
   - Parameter tuning experiments
   - Image preprocessing comparison
   - Confidence threshold testing
   - Complete pipeline with visualization

### Flask Web Application

1. **Start the web server:**
   ```bash
   source env-license-plate-ocr/bin/activate
   python flask_app.py
   ```

2. **Access the application** at: http://localhost:5000

3. **Upload license plate images** and see real-time OCR results with Pass/Fail validation

## Configuration

### OCR Parameters (Tunable)
- `contrast_ths`: 0.05 (contrast sensitivity)
- `text_threshold`: 0.6 (text detection threshold)
- `low_text`: 0.3 (faint text detection)
- `confidence_threshold`: 0.5 (minimum confidence for validation)

### Image Preprocessing
- **Contrast enhancement**: 1.25x multiplier
- **Unsharp masking**: radius=1.0, percent=120, threshold=3


## Lab Requirements Completed

### ✅ Required Tasks:
1. **Parameter Experimentation**: Multiple parameter sets tested with analysis
2. **Confidence Threshold**: Set to 0.5 with justification
4. **Visualization**: Bounding boxes with annotations and status indicators

### 📊 Key Findings:
- **Best Parameters**: contrast_ths=0.05, text_threshold=0.6, low_text=0.3
- **Preprocessing Impact**: 25% contrast boost + unsharp masking significantly improves accuracy
- **Confidence Threshold**: 0.5 provides good balance between accuracy and detection rate


## Technical Details

### OCR Pipeline:
1. **Load Image** → Convert to RGB
2. **Preprocess** → Enhance contrast + apply unsharp masking
3. **OCR Detection** → EasyOCR with Thai character support
4. **Confidence Filtering** → Remove low-confidence detections
6. **Visualization** → Draw results with Pass/Fail status

### Validation Logic:
- Confidence must exceed threshold (0.5)
- Results displayed with color coding (Green=Pass, Red=Fail)

## Future Enhancements

### Optional Extensions:
- **Cloud OCR Integration**: Google Vision API or AWS Textract
- **Real-time Processing**: Camera integration with live OCR
- **Database Storage**: PostgreSQL/MySQL for license plate records
- **Messaging Integration**: LINE Bot notifications for results
- **Edge Computing**: Optimize for Raspberry Pi deployment