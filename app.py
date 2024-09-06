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
    # MOPIC 글자를 크게 표시
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

# A_data 계산 함수
def calculate_physical_quantity(df, pressure):
    pressures = df.columns.astype(float)
    if pressure in pressures:
        return df[pressure]
    else:
        interp_function = interp1d(pressures, df.values, axis=1, fill_value="extrapolate")
        interpolated_values = interp_function(pressure)
        return pd.Series(interpolated_values, index=df.index)
 
# Overpressure 계산 함수
def calculate_overpressure(df, pressure, b_data_value):
    if pressure not in df.columns:
        interp_function_pressure = interp1d(df.columns.astype(float), df.values, axis=1, fill_value="extrapolate")
        interpolated_values = interp_function_pressure(pressure)
    else:
        interpolated_values = df[pressure].values
    
    interp_function_b_data = interp1d(df.index, interpolated_values, fill_value="extrapolate", bounds_error=False)
    overpressure = interp_function_b_data(b_data_value)
    
    return overpressure

# C_data 계산 함수 (Impulse 첫 번째 시트)
def calculate_c_data(df, pressure):
    pressures = df.columns.astype(float)
    if pressure in pressures:
        return df[pressure]
    else:
        interp_function = interp1d(pressures, df.values, axis=1, fill_value="extrapolate")
        return interp_function(pressure)
    
# D_data 계산 함수 (Impulse 두 번째 시트)
def calculate_d_data(df, volume, c_data_value):
    volumes = df.columns.astype(float)
    if volume in volumes:
        column_data = df[volume]
    else:
        interp_function = interp1d(volumes, df.values, axis=1, fill_value="extrapolate")
        column_data = pd.Series(interp_function(volume), index=df.index)
    
    interp_function_d = interp1d(df.index, column_data, fill_value="extrapolate", bounds_error=False)
    return interp_function_d(c_data_value)

# E_data 계산 함수 (Impulse 세 번째 시트)
def calculate_e_data(df, volume, d_data_value):
    volumes = df.columns.astype(float)
    if volume in volumes:
        column_data = df[volume]
    else:
        interp_function = interp1d(volumes, df.values, axis=1, fill_value="extrapolate")
        column_data = pd.Series(interp_function(volume), index=df.index)
    
    interp_function_e = interp1d(df.index, column_data, fill_value="extrapolate", bounds_error=False)
    return interp_function_e(d_data_value)

# Impulse 계산 함수 (Impulse 네 번째 시트)
def calculate_impulse(df, volume, e_data_value):
    volumes = df.columns.astype(float)
    if volume in volumes:
        column_data = df[volume]
    else:
        interp_function = interp1d(volumes, df.values, axis=1, fill_value="extrapolate")
        column_data = pd.Series(interp_function(volume), index=df.index)

    interp_function_impulse = interp1d(df.index, column_data, fill_value="extrapolate", bounds_error=False)
    return interp_function_impulse(e_data_value)

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
    
    progress_bar.progress(20)
    status_text.text("Calculating A_data...")

    # A_data 계산
    A_data = calculate_physical_quantity(df_first_sheet_overpressure, pressure_input)
    A_data[A_data <= 0] = np.nan
    
    progress_bar.progress(40)
    status_text.text("Calculating B_data...")

    # B_data 계산
    if volume_input in df_second_sheet_overpressure.columns:
        B_data = df_second_sheet_overpressure[volume_input]
    else:
        interp_function_B = interp1d(df_second_sheet_overpressure.columns.astype(float), df_second_sheet_overpressure.values, axis=1, fill_value="extrapolate")
        B_data = pd.Series(interp_function_B(volume_input), index=df_second_sheet_overpressure.index)

    interp_function_A = interp1d(df_second_sheet_overpressure.index, B_data, fill_value="extrapolate", bounds_error=False)
    B_data_interpolated = pd.Series(interp_function_A(A_data), index=A_data.index)
    B_data_interpolated[B_data_interpolated <= 0] = np.nan

    progress_bar.progress(60)
    status_text.text("Calculating Overpressure...")

    # Overpressure 계산
    overpressure_values = B_data_interpolated.apply(lambda x: calculate_overpressure(df_third_sheet_overpressure, pressure_input, x))

    progress_bar.progress(70)
    status_text.text("Calculating C_data...")

    # C_data 계산
    C_data = calculate_c_data(df_first_sheet_impulse, pressure_input)
    C_data[C_data <= 0] = np.nan

    progress_bar.progress(80)
    status_text.text("Calculating D_data, E_data, and Impulse...")

    # D_data, E_data, Impulse 계산
    D_data = [calculate_d_data(df_second_sheet_impulse, volume_input, c) for c in C_data]
    E_data = [calculate_e_data(df_third_sheet_impulse, volume_input, d) for d in D_data]
    Impulse_data = [calculate_impulse(df_fourth_sheet_impulse, volume_input, e) for e in E_data]

    progress_bar.progress(90)
    status_text.text("Finalizing data...")

    # NaN이 포함된 순서쌍 제거
    filtered_overpressure_values = overpressure_values.dropna()
    filtered_impulse_data = [i for i in Impulse_data if not np.isnan(i)]

    # 그래프 생성
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 첫 번째 그래프: Overpressure
    axs[0].plot(filtered_overpressure_values.index, filtered_overpressure_values, marker='o')
    axs[0].set_xscale('linear')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Distance (m)')
    axs[0].set_ylabel('Overpressure (kPa)')
    axs[0].set_title(f'{pressure_input}MPa, {volume_input}L Overpressure')

    # 두 번째 그래프: Impulse
    axs[1].plot(range(len(filtered_impulse_data)), filtered_impulse_data, marker='o')
    axs[1].set_xscale('linear')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Distance (m)')
    axs[1].set_ylabel('Impulse (kPa*s)')
    axs[1].set_title(f'{pressure_input}MPa, {volume_input}L Impulse')

    st.pyplot(fig)

    # 엑셀 파일로 저장
    output_file_path = 'output_data.xlsx'
    with BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            pd.DataFrame({
                'Distance (m)': filtered_overpressure_values.index,
                'Overpressure (kPa)': filtered_overpressure_values
            }).to_excel(writer, sheet_name='Overpressure', index=False)
            
            pd.DataFrame({
                'Distance (m)': range(len(filtered_impulse_data)),
                'Impulse (kPa*s)': filtered_impulse_data
            }).to_excel(writer, sheet_name='Impulse', index=False)
        
        buffer.seek(0)
        st.download_button(
            label="Download output File",
            data=buffer,
            file_name=output_file_path,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    progress_bar.progress(100)
    status_text.text("Calculation completed.")
    
    st.session_state.previous_results.append({
        'pressure': pressure_input,
        'volume': volume_input,
        'output_file_path': output_file_path,
        'graph': fig
    })
    
    if len(st.session_state.previous_results) > MAX_RECORDS:
        st.session_state.previous_results.pop(0)
    
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
