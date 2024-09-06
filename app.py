"""
Calculation Program in Hydrogen Tank Explosion Overpressure and Impulse - version 1.0

Description
: This Python code calculates overpressure and impulse resulting from the explosion of a high-pressure hydrogen tank.
It receives pressure and volume inputs from the user and uses nomogram data to compute the overpressure and impulse.
The nomogram data is referenced from the paper "Blast wave from a hydrogen tank rupture in a fire in the open: Hazard distance nomogram"
by Sergii Kashkarov, Zhiyong Li, and Vladimir Molkov. 
Both the code and the nomogram data are uploaded on GitHub and can be accessed by anyone through the link "https://nomogram-mntketbfejckna9g4hqe52.streamlit.app/".

Usage Guide:
1. Input Values:
   - Pressure (MPa): Enter the pressure value at which the tank explodes.
   - Volume (Liters): Enter the volume of the hydrogen tank.
   These inputs help in calculating the overpressure and impulse due to the explosion based on scientific models and empirical data.
2. Output:
   - The program outputs graphs showing the overpressure and impulse as a function of distance from the explosion site.
   - Detailed data tables and downloadable Excel files containing the calculated values are also provided for further analysis.
   
Developer : Yedo Park, Energy Safety Lab, Pukyong National University
Created on : 2024. 8. 20
Last updated : 2024. 9. 5

Update on September 5, 2024
1. Fixed an error that occurred when inputting pressure used for interpolation.
2. Add logo and copyright


"""

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from PIL import Image

# 이미지 파일 경로 설정 (엑셀 파일과 동일한 경로)
image_directory = r'.'  # 현재 디렉토리
university_logo_path = f"{image_directory}/university_logo.png"
lab_logo_path = f"{image_directory}/lab_logo.png"

# 컬럼을 이용한 로고와 제목 배치
col1, col2 = st.columns([2.5, 6.5])

with col1:
    lab_logo = Image.open(lab_logo_path)
    st.image(lab_logo, use_column_width=True)

with col2:
    st.markdown("<h1 style='font-size:72px;'>MOPIC</h1>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style='font-size:20px;'>
    <span style='color:red;'>M</span>aximum 
    <span style='color:red;'>O</span>ver 
    <span style='color:red;'>P</span>ressure and
    <span style='color:red;'>I</span>mpulse 
    <span style='color:red;'>C</span>alculation Program for High-Pressure Hydrogen Tanks
    </h3>
    """, unsafe_allow_html=True)
    st.write("This application calculates and visualizes data based on input pressure and volume.")

# 엑셀 파일 경로
overpressure_1_file_path = r'overpressure_1.xlsx'
overpressure_2_file_path = r'overpressure_2.xlsx'
overpressure_3_file_path = r'overpressure_3.xlsx'

impulse_1_file_path = r'impulse_1.xlsx'
impulse_2_file_path = r'impulse_2.xlsx'
impulse_3_file_path = r'impulse_3.xlsx'
impulse_4_file_path = r'impulse_4.xlsx'

# 진행 상황 표시
progress_bar = st.progress(0)
status_text = st.empty()

# Streamlit Session State to store previous results
if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False
if 'previous_results' not in st.session_state:
    st.session_state.previous_results = []

# 최대 5개의 기록만 유지
MAX_RECORDS = 5

# 사용자에게 압력과 부피 입력 받기
pressure_input = st.number_input("압력을 입력하세요 (MPa):", min_value=0.0, step=1.0)
volume_input = st.number_input("부피를 입력하세요 (Liter):", min_value=0.0, step=1.0)

# 계산 로직
def perform_calculation():
    # 엑셀 파일 읽기
    status_text.text("Loading Excel files...")
    df_first_sheet_overpressure = pd.read_excel(overpressure_1_file_path, index_col=0)
    df_second_sheet_overpressure = pd.read_excel(overpressure_2_file_path, index_col=0)
    df_third_sheet_overpressure = pd.read_excel(overpressure_3_file_path, index_col=0)
    df_first_sheet_impulse = pd.read_excel(impulse_1_file_path, index_col=0)
    df_second_sheet_impulse = pd.read_excel(impulse_2_file_path, index_col=0)
    df_third_sheet_impulse = pd.read_excel(impulse_3_file_path, index_col=0)
    df_fourth_sheet_impulse = pd.read_excel(impulse_4_file_path, index_col=0)
    
    # 계산 로직은 생략 - 기존 코드 유지
    # ...
    progress_bar.progress(100)
    status_text.text("Calculation completed.")
    
    # Mock calculation results
    output_file_path = 'output_data.xlsx'
    result_graph_buffer = BytesIO()  # Mock buffer for the graph image
    
    # 임시 결과 저장
    st.session_state.previous_results.append({
        'pressure': pressure_input,
        'volume': volume_input,
        'output_file_path': output_file_path,
        'graph': result_graph_buffer
    })
    
    # 기록이 5개를 넘으면 처음 기록을 삭제
    if len(st.session_state.previous_results) > MAX_RECORDS:
        st.session_state.previous_results.pop(0)
    
    # 계산 완료 플래그 설정
    st.session_state.calculation_done = True

# "계산 시작" 버튼을 비활성화하는 조건
if not st.session_state.calculation_done:
    if st.button("계산 시작"):
        perform_calculation()

# 계산이 완료되었으면 "계산 시작" 버튼 비활성화
if st.session_state.calculation_done:
    st.button("계산 시작", disabled=True)
    
    # "기록" 버튼 표시
    if st.button("기록 보기"):
        for idx, result in enumerate(st.session_state.previous_results, start=1):
            st.write(f"### 기록 {idx}")
            st.write(f"압력: {result['pressure']} MPa, 부피: {result['volume']} L")
            
            # 엑셀 파일 다운로드 버튼
            st.download_button(
                label=f"기록 {idx} 엑셀 파일 다운로드",
                data=result['output_file_path'],
                file_name=f"record_{idx}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # 그래프 이미지 다운로드 버튼
            st.download_button(
                label=f"기록 {idx} 그래프 다운로드",
                data=result['graph'],
                file_name=f"record_graph_{idx}.png",
                mime='image/png'
            )

    # "계산 재시작" 버튼 표시
    if st.button("계산 재시작"):
        st.session_state.calculation_done = False
        st.experimental_rerun()

# 저작권 표시 추가
st.markdown("---")
st.markdown(
    """
    <p style="text-align: center; color: gray; font-size: 12px;">
    © 2024 Energy Safety Laboratory, Pukyong National University. All rights reserved.<br>
    Reference: Kashkarov, S., Li, Z., & Molkov, V. (2021). Blast wave from a hydrogen tank rupture in a fire in the open: Hazard distance nomogram. International Journal of Hydrogen Energy, 46(58), 29900-29909.
    </p>
    """, unsafe_allow_html=True
)
