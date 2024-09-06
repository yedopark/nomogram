import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import requests
from PIL import Image

# 이미지 파일 경로 설정 (엑셀 파일과 동일한 경로)
image_directory = r'.'  # 현재 디렉토리
university_logo_path = f"{image_directory}/university_logo.png"
lab_logo_path = f"{image_directory}/lab_logo.png"

# 컬럼을 이용한 로고와 제목 배치
col1, col2 = st.columns([2.5, 6.5])

with col1:
    # 연구실 로고 표시
    lab_logo = Image.open(lab_logo_path)
    st.image(lab_logo, use_column_width=True)

with col2:
    # 중앙의 제목
    st.markdown("<h1 style='font-size:72px;'>MOPIC</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style='font-size:20px;'>
    <span style='color:red;'>M</span>aximum 
    <span style='color:red;'>O</span>ver<span style='color:red;'>P</span>ressure and
    <span style='color:red;'>I</span>mpulse 
    <span style='color:red;'>C</span>alculation Program for High-Pressure Hydrogen Tanks Explosion
    </h3>
    """, unsafe_allow_html=True)
    
    st.write("This application calculates and visualizes data based on input pressure and volume.")

# Streamlit Session State to store previous results
if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False
if 'previous_results' not in st.session_state:
    st.session_state.previous_results = []
if 'previous_inputs' not in st.session_state:
    st.session_state.previous_inputs = []

# 사용자에게 압력과 부피 입력 받기
st.markdown("<h3 style='font-size:24px;'>Enter Pressure (MPa):</h3>", unsafe_allow_html=True)
pressure_input = st.number_input("", min_value=0.0, step=1.0, key="pressure_input")

st.markdown("<h3 style='font-size:24px;'>Enter Volume (Liters):</h3>", unsafe_allow_html=True)
volume_input = st.number_input("", min_value=0.0, step=1.0, key="volume_input")


# 계산 완료 시 상태 초기화
def clear_calculation_state():
    st.session_state.calculation_done = False

# 엑셀 파일 경로 (분리된 파일 경로로 수정)
overpressure_1_file_path = r'overpressure_1.xlsx'
overpressure_2_file_path = r'overpressure_2.xlsx'
overpressure_3_file_path = r'overpressure_3.xlsx'

impulse_1_file_path = r'impulse_1.xlsx'
impulse_2_file_path = r'impulse_2.xlsx'
impulse_3_file_path = r'impulse_3.xlsx'
impulse_4_file_path = r'impulse_4.xlsx'

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

# C_data 계산 함수
def calculate_c_data(df, pressure):
    pressures = df.columns.astype(float)
    if pressure in pressures:
        return df[pressure]
    else:
        interp_function = interp1d(pressures, df.values, axis=1, fill_value="extrapolate")
        return interp_function(pressure)

# D_data 계산 함수
def calculate_d_data(df, volume, c_data_value):
    volumes = df.columns.astype(float)
    if volume in volumes:
        column_data = df[volume]
    else:
        interp_function = interp1d(volumes, df.values, axis=1, fill_value="extrapolate")
        column_data = pd.Series(interp_function(volume), index=df.index)
    
    interp_function_d = interp1d(df.index, column_data, fill_value="extrapolate", bounds_error=False)
    return interp_function_d(c_data_value)

# E_data 계산 함수
def calculate_e_data(df, volume, d_data_value):
    volumes = df.columns.astype(float)
    if volume in volumes:
        column_data = df[volume]
    else:
        interp_function = interp1d(volumes, df.values, axis=1, fill_value="extrapolate")
        column_data = pd.Series(interp_function(volume), index=df.index)
    
    interp_function_e = interp1d(df.index, column_data, fill_value="extrapolate", bounds_error=False)
    return interp_function_e(d_data_value)

# Impulse 계산 함수
def calculate_impulse(df, volume, e_data_value):
    try:
        volumes = df.columns.astype(float)
        if volume in volumes:
            column_data = df[volume]
        else:
            interp_function = interp1d(volumes, df.values, axis=1, fill_value="extrapolate")
            column_data = pd.Series(interp_function(volume), index=df.index)

        interp_function_impulse = interp1d(df.index, column_data, fill_value="extrapolate", bounds_error=False)
        result = interp_function_impulse(e_data_value)

        if np.isnan(result).any():
            return np.nan  # NaN이 포함된 경우 경고 없이 NaN 반환

        return round(float(result), 3)
    except Exception as e:
        st.error(f"Error in calculating impulse: {e}")
        return np.nan

# 진행 상황 표시
progress_bar = st.progress(0)
status_text = st.empty()

if not st.session_state.calculation_done:
    if st.button("Start Calculation"):
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
        A_data[A_data <= 0] = np.nan  # A_data에서 0 또는 음수 값을 NaN으로 변환
        
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
        B_data_interpolated[B_data_interpolated <= 0] = np.nan  # B_data에서 0 또는 음수 값을 NaN으로 변환

        progress_bar.progress(60)
        status_text.text("Calculating Overpressure...")

        # 세 번째 시트에서 overpressure 계산
        overpressure_values = B_data_interpolated.apply(lambda x: calculate_overpressure(df_third_sheet_overpressure, pressure_input, x))

        progress_bar.progress(70)
        status_text.text("Calculating C_data...")

        # C_data 계산
        C_data = calculate_c_data(df_first_sheet_impulse, pressure_input)
        C_data[C_data <= 0] = np.nan  # C_data에서 0 또는 음수 값을 NaN으로 변환

        progress_bar.progress(80)
        status_text.text("Calculating D_data, E_data, and Impulse...")

        # D_data 계산
        D_data = [calculate_d_data(df_second_sheet_impulse, volume_input, c) for c in C_data]

        # E_data 계산
        E_data = [calculate_e_data(df_third_sheet_impulse, volume_input, d) for d in D_data]

        # Impulse 계산
        Impulse_data = []
        for e in E_data:
            Impulse_data.append(calculate_impulse(df_fourth_sheet_impulse, volume_input, e))

        progress_bar.progress(90)
        status_text.text("Finalizing data...")
        
        # 결과 데이터 정리
        output_df_overpressure = pd.DataFrame({
            'Distance (m)': df_first_sheet_overpressure.index,
            'Overpressure (kPa)': overpressure_values
        })

        output_df_impulse = pd.DataFrame({
            'Distance_2 (m)': df_first_sheet_impulse.index,
            'Impulse (kPa*s)': Impulse_data
        })

        # 엑셀 파일로 저장 (저장한 후 다운로드 버튼 추가)
        output_file_path = 'output_data.xlsx'
        with BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                output_df_overpressure.to_excel(writer, sheet_name='Overpressure Data', index=False)
                output_df_impulse.to_excel(writer, sheet_name='Impulse Data', index=False)
            buffer.seek(0)
            st.download_button(
                label="Download Result Excel File",
                data=buffer,
                file_name=output_file_path,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # 결과 그래프 그리기 (y축 값이 0보다 작거나 같을 때 표시하지 않도록 필터링)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Overpressure 그래프 데이터에서 0 이하의 값 제거
        filtered_overpressure_df = output_df_overpressure[output_df_overpressure['Overpressure (kPa)'] > 0]

        # Impulse 그래프 데이터에서 0 이하의 값 제거
        filtered_impulse_df = output_df_impulse[output_df_impulse['Impulse (kPa*s)'] > 0]


        # 첫 번째 그래프: Overpressure
        axs[0].plot(output_df_overpressure['Distance (m)'], output_df_overpressure['Overpressure (kPa)'], marker='o', linestyle='-')
        axs[0].set_xscale('linear')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Distance (m)')
        axs[0].set_ylabel('Overpressure (kPa)')
        axs[0].set_title(f'{pressure_input}MPa, {volume_input}L ')

        # 두 번째 그래프: Impulse
        axs[1].plot(output_df_impulse['Distance_2 (m)'], output_df_impulse['Impulse (kPa*s)'], marker='o', linestyle='-')
        axs[1].set_xscale('linear')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Distance (m)')
        axs[1].set_ylabel('Impulse (kPa*s)')
        axs[1].set_title(f'{pressure_input}MPa, {volume_input}L ')

        st.pyplot(fig)

        # 그래프 이미지 다운로드 버튼 추가
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        st.download_button('Download Graph Image', buffer, file_name='graph.png', mime='image/png')

        # 계산 완료 상태 저장
        st.session_state.previous_results.append({
            'pressure': pressure_input,
            'volume': volume_input,
            'output_file_path': output_file_path,
            'graph': buffer
        })
        st.session_state.previous_inputs = [pressure_input, volume_input]
        st.session_state.calculation_done = True

        progress_bar.progress(100)
        status_text.text("Calculation completed.")

# 계산 완료 시 "계산 완료" 버튼과 "처음으로" 버튼을 보여줌
if st.session_state.calculation_done:
    if st.button("Start Over"):
        clear_calculation_state()  # 계산 상태만 초기화

# 이전에 저장된 결과 엑셀 파일과 그래프 보기
for idx, result in enumerate(st.session_state.previous_results, start=1):
    st.write(f"### Previous Results {idx}")
    st.write(f"Pressure: {result['pressure']} MPa, Volume: {result['volume']} L")
    
    # 엑셀 파일 다운로드 버튼
    st.download_button(
        label=f"Download Previous Results {idx} Excel File",
        data=result['output_file_path'],
        file_name=f"previous_result_{idx}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # 그래프 이미지 다운로드 버튼
    st.download_button(
        label=f"Download Previous Results {idx} Graph image",
        data=result['graph'],
        file_name=f"previous_graph_{idx}.png",
        mime='image/png'
    )

# 저작권 표시 추가
st.markdown("---")  # 구분선을 추가하여 시각적으로 구분
st.markdown(
    """
    <p style="text-align: center; color: gray; font-size: 12px;">
    © 2024 Energy Safety Laboratory, Pukyong National University. All rights reserved.<br>
    Reference: Kashkarov, S., Li, Z., & Molkov, V. (2021). Blast wave from a hydrogen tank rupture in a fire in the open: Hazard distance nomogram. International Journal of Hydrogen Energy, 46(58), 29900-29909.
    </p>
    """, unsafe_allow_html=True
)
