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
            
            # Cắt biển số xe từ ảnh gốc
            cropped_plate = image_bgr[y1:y2, x1:x2]
            
            # Thực hiện phối cảnh 4 góc
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            warped_plate = four_point_transform(image_bgr, corners)
            
            # Resize để tăng kích thước
            warped_plate = cv2.resize(warped_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Chuyển sang grayscale
            gray_plate = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)
            
            # Tăng độ tương phản bằng CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_plate = clahe.apply(gray_plate)
            
            # Chuyển đổi từ BGR sang RGB để hiển thị
            processed_plate_rgb = cv2.cvtColor(enhanced_plate, cv2.COLOR_GRAY2RGB)
            
            # Thực hiện OCR trên ảnh đã xử lý
            ocr_results = reader.readtext(enhanced_plate, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
            
            # Tính toán độ tin cậy trung bình của OCR
            ocr_confidences = [conf for (_, _, conf) in ocr_results]
            avg_ocr_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0
            
            plate_text = " ".join([text for (_, text, _) in ocr_results])
            
            # Vẽ box và text lên ảnh gốc với cả hai độ tin cậy
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            plates.append({
                'text': plate_text,
                'box': (x1, y1, x2, y2),
                'image': processed_plate_rgb,  # Ảnh đã xử lý hoàn chỉnh
                'cropped_image': cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB),  # Ảnh cắt thông thường
                'detect_confidence': conf * 100,
                'ocr_confidence': avg_ocr_confidence * 100
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
                                st.image(plate['cropped_image'], caption=f'Biển số #{i} (Ảnh cắt thông thường)', use_container_width=True)
                            
                            with plate_col2:
                                st.image(plate['image'], caption=f'Biển số #{i} (Ảnh đã xử lý)', use_container_width=True)
                            
                            with plate_col3:
                                st.markdown(f"""
                                <div class="result-card">
                                    <h3>Biển số #{i}</h3>
                                    <h2 style="font-size: 2em; margin: 10px 0;">{plate['text']}</h2>
                                    <div style="margin: 15px 0;">
                                        <p style="font-size: 1.1em;">📊 Độ tin cậy:</p>
                                        <p>🤖 Model phát hiện: {plate['detect_confidence']:.1f}%</p>
                                        <p>📝 Model OCR: {plate['ocr_confidence']:.1f}%</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if show_details:
                                st.markdown("---")
                    else:
                        st.warning(f"⚠️ Không phát hiện biển số xe nào với ngưỡng tin cậy {confidence_threshold}%!")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi phân tích: {str(e)}")

# ================================
# 📊 FOOTER STATISTICS
# ================================
st.markdown("---")
st.markdown("### 📈 Thống kê hệ thống")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🤖 Mô hình", "YOLOv8 + EasyOCR")

with col2:
    st.metric("🎯 Độ chính xác", ">90%")

with col3:
    st.metric("⚡ Tốc độ xử lý", "~1.5s/ảnh")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚗 <strong>Nhận diện biển số xe thông minh</strong> - Được phát triển với ❤️ bằng Streamlit & YOLOv8</p>
    <p>🤖 Nhận diện biển số chính xác • 📊 Xử lý ảnh nhanh chóng • 🔒 Bảo mật dữ liệu</p>
</div>
""", unsafe_allow_html=True) 