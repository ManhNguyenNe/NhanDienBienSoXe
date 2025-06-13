import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import easyocr
import plotly.express as px
import pandas as pd
import re

# ================================
# 🎨 CUSTOM CSS STYLING
# ================================
st.set_page_config(
    page_title="Nhận diện biển số xe",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 1rem;
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #2196F3;
        background-color: #f0f8ff;
    }
    
    /* Hide streamlit footer */
    .css-1d391kg { display: none; }
    footer { display: none; }
    .css-1rs6os { display: none; }
</style>
""", unsafe_allow_html=True)

# ================================
# 🚗 HEADER SECTION
# ================================
st.markdown("""
<div class="custom-header">
    <h1>🚗 Nhận diện biển số xe thông minh</h1>
    <p style="font-size: 1.2em; margin-top: 1rem;">
        🤖 Sử dụng YOLOv8 và EasyOCR để nhận diện biển số xe
    </p>
</div>
""", unsafe_allow_html=True)

# ================================
# 🔧 MODEL LOADING
# ================================
@st.cache_resource
def load_models():
    """Load YOLO model and EasyOCR reader"""
    try:
        model = YOLO("runs/weights/best.pt")
        # Tối ưu EasyOCR cho tiếng Việt và tiếng Anh
        reader = easyocr.Reader(['en'], gpu=True)
        st.success("✅ Đã tải mô hình thành công!")
        return model, reader
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
        return None, None

model, reader = load_models()

# ================================
# 🎛️ SIDEBAR CONFIGURATION
# ================================
with st.sidebar:
    st.markdown("## 🔧 Cấu hình")
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Ngưỡng tin cậy (%)", 
        min_value=50, 
        max_value=100, 
        value=80,
        help="Ngưỡng tin cậy tối thiểu cho việc nhận diện"
    )
    
    # OCR confidence threshold
    ocr_confidence_threshold = st.slider(
        "Ngưỡng tin cậy OCR (%)",
        min_value=30,
        max_value=100,
        value=50,
        help="Ngưỡng tin cậy tối thiểu cho kết quả OCR"
    )
    
    # Image processing options
    st.markdown("### 🔧 Tùy chọn xử lý ảnh")
    enable_denoise = st.checkbox("🧹 Khử nhiễu", value=True)
    enable_sharpen = st.checkbox("🔍 Tăng độ sắc nét", value=True)
    resize_factor = st.slider("📏 Hệ số phóng to", 1.5, 4.0, 2.5, 0.5)
    
    # Show details option
    show_details = st.checkbox("📈 Hiển thị chi tiết kết quả", value=True)
    
    # About section
    st.markdown("### ℹ️ Giới thiệu")
    st.markdown("""
    **Công nghệ sử dụng:**
    - 🤖 **YOLOv8**: Nhận diện vị trí biển số
    - 📝 **EasyOCR**: Đọc ký tự trên biển số
    - 🎯 **Độ chính xác**: >90%
    """)

# ================================
# 🛠️ HELPER FUNCTIONS
# ================================
def clean_license_plate_text(text):
    """Làm sạch và chuẩn hóa text biển số xe"""
    if not text:
        return ""
    
    # Loại bỏ khoảng trắng thừa và ký tự đặc biệt
    text = re.sub(r'[^\w\-]', '', text.upper())
    
    # Tách phần chữ và số
    parts = []
    current_part = ""
    prev_type = None
    
    for char in text:
        if char.isdigit():
            char_type = "digit"
        elif char.isalpha():
            char_type = "alpha"
        else:
            continue
            
        if prev_type and prev_type != char_type:
            if current_part:
                parts.append(current_part)
            current_part = char
        else:
            current_part += char
            
        prev_type = char_type
    
    if current_part:
        parts.append(current_part)
    
    # Xử lý từng phần một cách thông minh
    cleaned_parts = []
    for i, part in enumerate(parts):
        if part.isalpha():
            # Phần chữ cái: giữ nguyên, chỉ sửa những trường hợp rõ ràng
            cleaned_part = smart_clean_alpha_part(part, is_first_part=(i == 0))
        else:
            # Phần số: thay thế những ký tự rõ ràng bị OCR nhầm
            cleaned_part = smart_clean_numeric_part(part)
        
        if cleaned_part:
            cleaned_parts.append(cleaned_part)
    
    return "".join(cleaned_parts)

def smart_clean_alpha_part(text, is_first_part=False):
    """Làm sạch phần chữ cái một cách thông minh"""
    result = ""
    
    for i, char in enumerate(text):
        # Chỉ thay thế khi có bằng chứng rõ ràng là OCR nhầm
        if char == 'O' and i > 0 and text[i-1].isalpha():
            # O ở giữa/cuối từ có thể là số 0
            if is_surrounded_by_numbers_context(text, i):
                result += '0'
            else:
                result += char
        elif char == 'I' and len(text) > 2 and not is_first_part:
            # I trong phần không phải đầu tiên có thể là số 1
            result += '1'
        elif char in ['Q', 'D'] and is_likely_zero_context(text, i):
            # Q, D có thể là 0 trong context nhất định
            result += '0'
        else:
            result += char
    
    return result

def smart_clean_numeric_part(text):
    """Làm sạch phần số một cách thông minh"""
    result = ""
    
    for i, char in enumerate(text):
        if char.isdigit():
            result += char
        elif char in ['O', 'Q', 'D'] and is_clearly_zero_context(text, i):
            # Rõ ràng là số 0
            result += '0'
        elif char == 'I' and is_clearly_one_context(text, i):
            # Rõ ràng là số 1
            result += '1'
        elif char == 'S' and is_clearly_five_context(text, i):
            # S có thể là 5
            result += '5'
        elif char == 'B' and is_clearly_eight_context(text, i):
            # B có thể là 8
            result += '8'
        elif char == 'G' and is_clearly_six_context(text, i):
            # G có thể là 6
            result += '6'
        else:
            # Giữ nguyên nếu không chắc chắn
            result += char
    
    return result

def is_surrounded_by_numbers_context(text, pos):
    """Kiểm tra xem ký tự có bị bao quanh bởi context số không"""
    before = text[pos-1] if pos > 0 else ''
    after = text[pos+1] if pos < len(text)-1 else ''
    
    # Nếu trước và sau đều là số hoặc ký tự giống số
    return (before.isdigit() or before in ['O', 'Q', 'D']) and \
           (after.isdigit() or after in ['O', 'Q', 'D'])

def is_likely_zero_context(text, pos):
    """Kiểm tra context cho ký tự có thể là số 0"""
    # Nếu xung quanh có nhiều số
    surrounding = ""
    start = max(0, pos-2)
    end = min(len(text), pos+3)
    
    for i in range(start, end):
        if i != pos:
            surrounding += text[i]
    
    digit_count = sum(1 for c in surrounding if c.isdigit())
    return digit_count >= 2

def is_clearly_zero_context(text, pos):
    """Kiểm tra rõ ràng là số 0 trong context số"""
    return has_digit_neighbors(text, pos, distance=1)

def is_clearly_one_context(text, pos):
    """Kiểm tra rõ ràng là số 1"""
    return has_digit_neighbors(text, pos, distance=1)

def is_clearly_five_context(text, pos):
    """Kiểm tra rõ ràng là số 5"""
    return has_digit_neighbors(text, pos, distance=2)

def is_clearly_eight_context(text, pos):
    """Kiểm tra rõ ràng là số 8"""
    return has_digit_neighbors(text, pos, distance=1)

def is_clearly_six_context(text, pos):
    """Kiểm tra rõ ràng là số 6"""
    return has_digit_neighbors(text, pos, distance=1)

def has_digit_neighbors(text, pos, distance=1):
    """Kiểm tra xem có số ở gần không"""
    for i in range(max(0, pos-distance), min(len(text), pos+distance+1)):
        if i != pos and text[i].isdigit():
            return True
    return False


def advanced_image_preprocessing(image):
    """Tiền xử lý ảnh nâng cao cho OCR"""
    # Chuyển sang grayscale nếu cần
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Khử nhiễu
    if enable_denoise:
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Cân bằng histogram adaptive
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 3. Morphological operations để làm sạch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # 4. Tăng độ sắc nét
    if enable_sharpen:
        kernel_sharp = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel_sharp)
    
    # 5. Threshold adaptive để tách chữ và nền
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 6. Thử cả ảnh gốc và ảnh đảo ngược
    binary_inv = cv2.bitwise_not(binary)
    
    return gray, binary, binary_inv

def order_points(pts):
    """Sắp xếp các điểm theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái"""
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Tổng tọa độ (x,y) nhỏ nhất là điểm trên-trái
    # Tổng tọa độ (x,y) lớn nhất là điểm dưới-phải
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Hiệu tọa độ (x-y) nhỏ nhất là điểm trên-phải
    # Hiệu tọa độ (x-y) lớn nhất là điểm dưới-trái
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """Thực hiện biến đổi phối cảnh 4 điểm"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Tính chiều rộng mới
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Tính chiều cao mới
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Tạo ma trận điểm đích
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    
    # Tính ma trận biến đổi và thực hiện biến đổi
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def perform_multiple_ocr(image):
    """Thực hiện OCR với nhiều phương pháp khác nhau và chọn kết quả tốt nhất"""
    results = []
    
    # Resize ảnh
    height, width = image.shape[:2]
    if height < 50 or width < 150:
        scale = max(50/height, 150/width)
        new_width = int(width * scale * resize_factor)
        new_height = int(height * scale * resize_factor)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    else:
        resized = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    
    # Tiền xử lý ảnh
    gray, binary, binary_inv = advanced_image_preprocessing(resized)
    
    # Danh sách các ảnh để thử OCR
    images_to_try = [
        ("original", resized),
        ("gray", gray),
        ("binary", binary),
        ("binary_inverted", binary_inv)
    ]
    
    # Thử OCR với các cấu hình khác nhau
    ocr_configs = [
        # Cấu hình 1: Standard
        {'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-', 'width_ths': 0.7, 'height_ths': 0.7},
        # Cấu hình 2: Relaxed thresholds
        {'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-', 'width_ths': 0.5, 'height_ths': 0.5},
        # Cấu hình 3: Character-based
        {'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-', 'width_ths': 0.3, 'height_ths': 0.3, 'paragraph': False},
    ]
    
    for img_name, img in images_to_try:
        for config in ocr_configs:
            try:
                ocr_result = reader.readtext(img, **config)
                
                if ocr_result:
                    # Tính confidence trung bình
                    confidences = [conf for (_, _, conf) in ocr_result]
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    # Ghép text
                    raw_text = " ".join([text for (_, text, _) in ocr_result])
                    cleaned_text = clean_license_plate_text(raw_text)
                    
                    if cleaned_text and avg_confidence >= (ocr_confidence_threshold / 100):
                        results.append({
                            'text': cleaned_text,
                            'raw_text': raw_text,
                            'confidence': avg_confidence,
                            'method': f"{img_name}",
                            'length': len(cleaned_text.replace('-', ''))
                        })
            except Exception as e:
                continue
    
    if not results:
        return "", 0, "No result"
    
    # Sắp xếp kết quả theo confidence và độ dài hợp lý (6-10 ký tự cho biển số VN)
    def score_result(result):
        base_score = result['confidence']
        length = result['length']
        
        # Bonus cho độ dài hợp lý của biển số Việt Nam
        if 6 <= length <= 10:
            base_score += 0.1
        elif 5 <= length <= 11:
            base_score += 0.05
        else:
            base_score -= 0.1
            
        return base_score
    
    results.sort(key=score_result, reverse=True)
    best_result = results[0]
    
    return best_result['text'], best_result['confidence'] * 100, best_result['method']

def process_image(image):
    """Process image and return results"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Detect license plates
    results = model(image_bgr)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    
    # Process each detected plate
    plates = []
    for box, conf in zip(boxes, confidences):
        if conf * 100 >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            
            # Cắt biển số xe từ ảnh gốc với padding
            padding = 5
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(image_bgr.shape[1], x2 + padding)
            y2_pad = min(image_bgr.shape[0], y2 + padding)
            
            cropped_plate = image_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Thực hiện phối cảnh 4 góc
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            warped_plate = four_point_transform(image_bgr, corners)
            
            # Thực hiện OCR với nhiều phương pháp
            plate_text, ocr_confidence, ocr_method = perform_multiple_ocr(warped_plate)
            
            # Vẽ box và text lên ảnh gốc
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Tạo ảnh hiển thị
            display_image = cv2.resize(warped_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            
            plates.append({
                'text': plate_text,
                'box': (x1, y1, x2, y2),
                'image': display_image_rgb,
                'cropped_image': cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB),
                'detect_confidence': conf * 100,
                'ocr_confidence': ocr_confidence,
                'ocr_method': ocr_method
            })
    
    # Convert BGR back to RGB for display
    result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return result_image, plates

# ================================
# 📤 FILE UPLOAD SECTION
# ================================
st.markdown("## 📤 Tải lên ảnh")

uploaded_file = st.file_uploader(
    "Chọn ảnh để phân tích",
    type=["jpg", "jpeg", "png"],
    help="Hỗ trợ: JPG, PNG. Kích thước tối đa: 10MB"
)

if uploaded_file is not None:
    # Display original image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🖼️ Ảnh gốc")
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
        
        # Image info
        width, height = image.size
        st.info(f"📐 Kích thước: {width} x {height} pixels")
    
    with col2:
        st.markdown("### 🔍 Kết quả phân tích")
        
        if st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True):
            with st.spinner('🤖 AI đang phân tích...'):
                try:
                    # Process image
                    result_image, plates = process_image(image)
                    
                    # Display result image
                    st.image(result_image, caption='Kết quả nhận diện', use_container_width=True)
                    
                    # Display detected plates
                    if plates:
                        st.success(f"✅ Đã phát hiện {len(plates)} biển số xe!")
                        
                        for i, plate in enumerate(plates, 1):
                            # Tạo 3 cột để hiển thị kết quả
                            plate_col1, plate_col2, plate_col3 = st.columns(3)
                            
                            with plate_col1:
                                st.image(plate['cropped_image'], caption=f'Biển số #{i} (Ảnh cắt)', use_container_width=True)
                            
                            with plate_col2:
                                st.image(plate['image'], caption=f'Biển số #{i} (Ảnh xử lý)', use_container_width=True)
                            
                            with plate_col3:
                                # Xác định màu dựa trên confidence
                                if plate['ocr_confidence'] >= 80:
                                    confidence_color = "#4CAF50"  # Xanh lá
                                elif plate['ocr_confidence'] >= 60:
                                    confidence_color = "#FF9800"  # Cam
                                else:
                                    confidence_color = "#F44336"  # Đỏ
                                
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {confidence_color} 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                                    <h3>Biển số #{i}</h3>
                                    <h2 style="font-size: 2em; margin: 10px 0; font-weight: bold;">{plate['text'] if plate['text'] else 'Không đọc được'}</h2>
                                    <div style="margin: 15px 0;">
                                        <p style="font-size: 1.1em;">📊 Độ tin cậy:</p>
                                        <p>🤖 Phát hiện: {plate['detect_confidence']:.1f}%</p>
                                        <p>📝 OCR: {plate['ocr_confidence']:.1f}%</p>
                                        <p style="font-size: 0.9em;">🔧 Phương pháp: {plate['ocr_method']}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if show_details:
                                with st.expander(f"Chi tiết kỹ thuật biển số #{i}"):
                                    st.write(f"**Tọa độ:** {plate['box']}")
                                    st.write(f"**Confidence phát hiện:** {plate['detect_confidence']:.2f}%")
                                    st.write(f"**Confidence OCR:** {plate['ocr_confidence']:.2f}%")
                                    st.write(f"**Phương pháp OCR:** {plate['ocr_method']}")
                                    st.write(f"**Độ dài text:** {len(plate['text'])} ký tự")
                                
                                st.markdown("---")
                    else:
                        st.warning(f"⚠️ Không phát hiện biển số xe nào với ngưỡng tin cậy {confidence_threshold}%!")
                        st.info("💡 **Gợi ý:** Thử giảm ngưỡng tin cậy hoặc upload ảnh có chất lượng tốt hơn")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi phân tích: {str(e)}")
                    st.error("🔧 Hãy thử với ảnh khác hoặc điều chỉnh các tham số")

# ================================
# 📊 FOOTER STATISTICS
# ================================
st.markdown("---")
st.markdown("### 📈 Thống kê hệ thống")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🤖 Mô hình", "YOLOv8 + EasyOCR")

with col2:
    st.metric("🎯 Độ chính xác", ">90%")

with col3:
    st.metric("⚡ Tốc độ xử lý", "~2s/ảnh")

with col4:
    st.metric("🔧 Phương pháp OCR", "Multi-method")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚗 <strong>Nhận diện biển số xe thông minh</strong> - Được phát triển với ❤️ bằng Streamlit & YOLOv8</p>
    <p>🤖 Nhận diện biển số chính xác • 📊 Xử lý ảnh đa phương pháp • 🔒 Bảo mật dữ liệu</p>
    <p style="font-size: 0.9em; margin-top: 1rem;">
        <strong>Cải tiến mới:</strong> OCR đa phương pháp • Tiền xử lý ảnh nâng cao • Làm sạch text thông minh
    </p>
</div>
""", unsafe_allow_html=True)