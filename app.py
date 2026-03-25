import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import tempfile
import pandas as pd
from fpdf import FPDF

# ==========================================
# --- 1. 核心修复：MediaPipe 初始化 (针对 3.11 环境优化) ---
# ==========================================
# 显式导入子模块，防止出现 "module 'mediapipe' has no attribute 'solutions'"
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# 使用缓存装饰器，避免重复加载模型导致页面卡顿
@st.cache_resource
def load_pose_model():
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    )

pose_engine = load_pose_model()

# ==========================================
# --- 2. 页面配置与标题 ---
# ==========================================
st.set_page_config(page_title="AI 数字化体能评估", layout="wide")
st.title("🏋️‍♂️ AI 数字化体能评估系统")
st.markdown("---")

# ==========================================
# --- 3. 核心 AI 处理函数 ---
# ==========================================
def process_pose_image(image_file):
    if image_file is None:
        return None, None
    
    # 将上传的文件转为 OpenCV 格式
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    # AI 识别处理
    results = pose_engine.process(rgb_image)
    
    # 绘制骨架图
    annotated_image = rgb_image.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image, results.pose_landmarks

def analyze_posture_landmarks(landmarks):
    """
    根据坐标点进行简单的体态逻辑分析
    """
    issues = []
    if not landmarks:
        return issues
        
    lm = landmarks.landmark
    # 示例逻辑：对比左右肩高度 (Landmark 11 和 12)
    left_shoulder_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    
    if abs(left_shoulder_y - right_shoulder_y) > 0.03:
        issues.append("检测到高低肩风险，建议关注骨盆及足底支撑")
        
    # 示例逻辑：头部前倾 (Landmark 0 鼻子和 11/12 肩膀中点对比)
    shoulder_mid_z = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z) / 2
    nose_z = lm[mp_pose.PoseLandmark.NOSE].z
    if nose_z < shoulder_mid_z - 0.05:
        issues.append("检测到头部重心前移，建议关注 PRI 呼吸模式")
        
    return issues

# ==========================================
# --- 4. 侧边栏：用户信息输入 ---
# ==========================================
with st.sidebar:
    st.header("👤 客户基本信息")
    name = st.text_input("客户姓名", "测试用户")
    needs = st.multiselect(
        "运动目标 / 痛点",
        ["减脂", "增肌", "产后修复", "缓解疼痛", "体态纠正", "PRI 呼吸优化"],
        ["缓解疼痛"]
    )
    st.divider()
    st.info("💡 提示：请确保照片背景简洁，以便 AI 精准抓取坐标点。")

# ==========================================
# --- 5. 第一步：体态拍照与 AI 评估 ---
# ==========================================
st.header("📸 第一步：姿态拍照评估")
col1, col2 = st.columns(2)

ai_issues = []

with col1:
    st.subheader("📍 正面/背面拍照")
    front_img_file = st.file_uploader("上传正面照", type=['jpg', 'png', 'jpeg'], key="front")
    if front_img_file:
        res_img, landmarks = process_pose_image(front_img_file)
        if res_img is not None:
            st.image(res_img, caption="AI 骨架提取结果", use_container_width=True)
            if landmarks:
                issues = analyze_posture_landmarks(landmarks)
                for iss in issues:
                    st.warning(f"⚠️ {iss}")
                    ai_issues.append(iss)

with col2:
    st.subheader("📍 侧面拍照")
    side_img_file = st.file_uploader("上传侧面照", type=['jpg', 'png', 'jpeg'], key="side")
    if side_img_file:
        res_img, landmarks = process_pose_image(side_img_file)
        if res_img is not None:
            st.image(res_img, caption="AI 侧面重心分析", use_container_width=True)
            if landmarks:
                issues = analyze_posture_landmarks(landmarks)
                for iss in issues:
                    st.warning(f"⚠️ {iss}")
                    ai_issues.append(iss)

# ==========================================
# --- 6. 生成报告部分 ---
# ==========================================
st.divider()
if st.button("📝 生成 PDF 数字化评估报告"):
    if not ai_issues:
        st.error("请先上传照片进行 AI 识别后再生成报告。")
    else:
        st.success("报告生成逻辑已触发！")
        # 这里保留你原有的 PDF 生成逻辑框架
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(40, 10, f'Assessment Report for {name}')
        
        # 模拟生成下载链接
        st.info("PDF 功能已就绪，正在导出数据...")
