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
    st.title("Calculation Program in Hydrogen Tank Explosion Overpressure and Impulse")
    st.write("This application calculates and visualizes data based on input pressure and volume.")

# 엑셀 파일 경로 (분리된 파일 경로로 수정)
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

# A_data 계산
def calculate_physical_quantity(df, pressure):
    pressures = df.columns.astype(float)
    if pressure in pressures:
        return df[pressure]
    else:
        interp_function = interp1d(pressures, df.values, axis=1, fill_value="extrapolate")
        interpolated_values = interp_function(pressure)
        return pd.Series(interpolated_values, index=df.index)
 
# Overpressure 계산    
def calculate_overpressure(df, pressure, b_data_value):
    if pressure not in df.columns:
        interp_function_pressure = interp1d(df.columns.astype(float), df.values, axis=1, fill_value="extrapolate")
        interpolated_values = interp_function_pressure(pressure)
    else:
        interpolated_values = df[pressure].values
    
    interp_function_b_data = interp1d(df.index, interpolated_values, fill_value="extrapolate", bounds_error=False)
    overpressure = interp_function_b_data(b_data_value)
    
    return overpressure

# C_data 계산 (Impulse의 첫 번째 시트에서 압력에 해당하는 값)
def calculate_c_data(df, pressure):
    pressures = df.columns.astype(float)
    if pressure in pressures:
        return df[pressure]
    else:
        interp_function = interp1d(pressures, df.values, axis=1, fill_value="extrapolate")
        return interp_function(pressure)
    
# D_data 계산 (Impulse의 두 번째 시트에서 부피와 C_data에 해당하는 값)
def calculate_d_data(df, volume, c_data_value):
    volumes = df.columns.astype(float)
    if volume in volumes:
        column_data = df[volume]
    else:
        interp_function = interp1d(volumes, df.values, axis=1, fill_value="extrapolate")
        column_data = pd.Series(interp_function(volume), index=df.index)
    
    interp_function_d = interp1d(df.index, column_data, fill_value="extrapolate", bounds_error=False)
    return interp_function_d(c_data_value)

# E_data 계산 (Impulse의 세 번째 시트에서 부피와 D_data에 해당하는 값)
def calculate_e_data(df, volume, d_data_value):
    volumes = df.columns.astype(float)
    if volume in volumes:
        column_data = df[volume]
    else:
        interp_function = interp1d(volumes, df.values, axis=1, fill_value="extrapolate")
        column_data = pd.Series(interp_function(volume), index=df.index)
    
    interp_function_e = interp1d(df.index, column_data, fill_value="extrapolate", bounds_error=False)
    return interp_function_e(d_data_value)

# Impulse 계산 (Impulse의 네 번째 시트에서 부피와 E_data에 해당하는 값)
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

# Streamlit Session State to store previous results
if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False
if 'previous_results' not in st.session_state:
    st.session_state.previous_results = []
if 'previous_inputs' not in st.session_state:
    st.session_state.previous_inputs = []

# 사용자에게 압력과 부피 입력 받기
pressure_input = st.number_input("압력을 입력하세요 (MPa):", min_value=0.0, step=1.0)
volume_input = st.number_input("부피를 입력하세요 (Liter):", min_value=0.0, step=1.0)

# 부동소수점 비교 시 허용 오차 범위를 설정
tolerance = 0.001
if any(abs(pressure_input - p) < tolerance for p in [20.00, 35.00, 70.00, 100.00]):
    pressure_input += 0.000001

if st.button("계산 시작"):
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
    
    # 배열들의 길이를 동일하게 맞춤 (Overpressure 관련 데이터)
    min_length_overpressure = min(len(A_data), len(B_data_interpolated), len(overpressure_values))
    A_data = A_data.iloc[:min_length_overpressure]
    B_data_interpolated = B_data_interpolated.iloc[:min_length_overpressure]
    overpressure_values = overpressure_values.iloc[:min_length_overpressure]

    # 배열들의 길이를 동일하게 맞춤 (Impulse 관련 데이터)
    min_length_impulse = min(len(C_data), len(D_data), len(E_data), len(Impulse_data))
    C_data = C_data[:min_length_impulse]
    D_data = D_data[:min_length_impulse]
    E_data = E_data[:min_length_impulse]
    Impulse_data = Impulse_data[:min_length_impulse]

    # Overpressure 관련 데이터를 첫 번째 시트에 저장
    output_df_overpressure = pd.DataFrame({
        'Distance (m)': df_first_sheet_overpressure.index[:min_length_overpressure],
        #'A_data': A_data,
        #'B_data_interpolated': B_data_interpolated,
        'Overpressure (kPa)': overpressure_values
    })

    # Distance_2 (m) 배열 추가 (impulse_1.xlsx의 첫 번째 열 데이터를 저장)
    df_first_sheet_impulse = pd.read_excel(impulse_1_file_path, index_col=0)
    Distance_2 = df_first_sheet_impulse.index  # 첫 번째 열 데이터

    # Impulse 관련 데이터를 두 번째 시트에 저장 (Distance_2 사용)
    output_df_impulse = pd.DataFrame({
        'Distance_2 (m)': Distance_2[:min_length_impulse],
        #'C_data': C_data,
        #'D_data': D_data,
        #'E_data': E_data,
        'Impulse (kPa*s)': Impulse_data
    })

    # 엑셀 파일로 저장 (저장한 후 다운로드 버튼 추가)
    output_file_path = 'output_data.xlsx'

    # Create a buffer to store the Excel file in memory
    with BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            output_df_overpressure.to_excel(writer, sheet_name='Overpressure Data', index=False)
            output_df_impulse.to_excel(writer, sheet_name='Impulse Data', index=False)
        buffer.seek(0)  # Move the pointer to the beginning of the buffer
        # Add a download button
        st.download_button(
            label="Download output File",
            data=buffer,
            file_name=output_file_path,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # NaN이 포함된 순서쌍 제거 후 0 이하 값도 제외한 그래프 그리기
    filtered_output_df_overpressure = output_df_overpressure.dropna()
    filtered_output_df_overpressure = filtered_output_df_overpressure[filtered_output_df_overpressure['Overpressure (kPa)'] > 0]

    filtered_output_df_impulse = output_df_impulse.dropna()
    filtered_output_df_impulse = filtered_output_df_impulse[filtered_output_df_impulse['Impulse (kPa*s)'] > 0]

    # 그래프 생성
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

   # 부동소수점 비교 시 허용 오차 범위를 설정
    tolerance = 0.001
    if any(abs(pressure_input - p) < tolerance for p in [20.000001, 35.000001, 70.000001, 100.000001]):
       pressure_input -= 0.000001

    # 첫 번째 그래프: Overpressure (y축 로그 스케일)
    axs[0].plot(filtered_output_df_overpressure['Distance (m)'], filtered_output_df_overpressure['Overpressure (kPa)'], marker='o', linestyle='-')
    axs[0].set_xscale('linear')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Distance (m)')
    axs[0].set_ylabel('Overpressure (kPa)')
    axs[0].set_title(f'{pressure_input}MPa, {volume_input}L ')

    # 두 번째 그래프: Impulse (x축을 Distance_2 (m)로 설정)
    axs[1].plot(filtered_output_df_impulse['Distance_2 (m)'], filtered_output_df_impulse['Impulse (kPa*s)'], marker='o', linestyle='-')
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

    # 진행 상황을 100%로 업데이트
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
    st.session_state.previous_inputs = [pressure_input, volume_input]
        
    # 계산 완료 플래그 설정
    st.session_state.calculation_done = True
else:
    # "계산 재시작" 버튼과 이전 결과 표시
    st.write("계산이 완료되었습니다.")
    if st.button("계산 재시작"):
        st.session_state.calculation_done = False

    # 이전에 입력했던 압력, 부피 및 결과 보여주기
    if st.session_state.previous_inputs:
        st.write(f"이전 입력 값: 압력 = {st.session_state.previous_inputs[0]} MPa, 부피 = {st.session_state.previous_inputs[1]} L")
    
    # 이전에 저장된 결과 엑셀 파일과 그래프 보기
    for idx, result in enumerate(st.session_state.previous_results, start=1):
        st.write(f"### 이전 결과 {idx}")
        st.write(f"압력: {result['pressure']} MPa, 부피: {result['volume']} L")
        
        # 엑셀 파일 다운로드 버튼
        st.download_button(
            label=f"이전 결과 {idx} 엑셀 파일 다운로드",
            data=result['output_file_path'],
            file_name=f"previous_result_{idx}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # 그래프 이미지 다운로드 버튼
        st.download_button(
            label=f"이전 결과 {idx} 그래프 다운로드",
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
