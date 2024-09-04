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
Last updated : 2024. 8. 20

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

# 사용자에게 압력과 부피 입력 받기
pressure_input = st.number_input("압력을 입력하세요:", min_value=0.0, step=1.0)
volume_input = st.number_input("부피를 입력하세요:", min_value=0.0, step=1.0)

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

    # 첫 번째 시트에 저장할 데이터프레임 생성
    df_first_sheet = pd.DataFrame({
        'Distance (m)': df_first_sheet_overpressure.index[:len(A_data)],
        'A_data': A_data,
        'B_data': B_data_interpolated,
        'Overpressure (kPa)': overpressure_values
    })

    # 배열들의 길이를 동일하게 맞춤 (가장 짧은 길이에 맞추기)
    min_length_impulse = min(len(C_data), len(D_data), len(E_data), len(Impulse_data))

    C_data = C_data[:min_length_impulse]
    D_data = D_data[:min_length_impulse]
    E_data = E_data[:min_length_impulse]
    Impulse_data = Impulse_data[:min_length_impulse]

    # Distance 값도 같은 길이로 자르기
    distance_impulse = df_first_sheet_overpressure.index[:min_length_impulse]

    # 두 번째 시트에 저장할 데이터프레임 생성
    df_second_sheet = pd.DataFrame({
        'Distance (m)': distance_impulse,  # Distance는 길이를 맞춘 값 사용
        'C_data': C_data,
        'D_data': D_data,
        'E_data': E_data,
        'Impulse (kPa*s)': Impulse_data
    })



    # 엑셀 파일에 저장 (첫번째 시트와 두번째 시트에 각각 데이터 저장)
    output_file_path = 'output_pressure_volume_data_with_impulse.xlsx'

    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        df_first_sheet.to_excel(writer, sheet_name='Overpressure', index=False)
        df_second_sheet.to_excel(writer, sheet_name='Impulse', index=False)

    # 엑셀 파일을 다운로드할 수 있는 버튼 추가
    with open(output_file_path, 'rb') as f:
        st.download_button('Download Full Output Excel File', f, file_name=output_file_path)

    # 그래프 생성
    st.write("### Graphs")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 첫 번째 그래프: Overpressure (y축 로그 스케일)
    filtered_df_first = df_first_sheet[df_first_sheet['Overpressure (kPa)'] > 0]  # 0 이상의 데이터만 필터링
    axs[0].plot(filtered_df_first['Distance (m)'], filtered_df_first['Overpressure (kPa)'], marker='o', linestyle='-')
    axs[0].set_xscale('linear')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Distance (m)')
    axs[0].set_ylabel('Overpressure (kPa)')
    axs[0].set_title(f'{pressure_input}MPa, {volume_input}L ')

    # 두 번째 그래프: Impulse
    filtered_df_second = df_second_sheet[df_second_sheet['Impulse (kPa*s)'] > 0]  # 0 이상의 데이터만 필터링
    axs[1].plot(filtered_df_second['Distance (m)'], filtered_df_second['Impulse (kPa*s)'], marker='o', linestyle='-')
    axs[1].set_xscale('linear')
    axs[1].set_yscale('linear')
    axs[1].set_xlabel('Distance (m)')
    axs[1].set_ylabel('Impulse (kPa*s)')
    axs[1].set_title('Impulse vs Distance')

    st.pyplot(fig)

    # 그래프 이미지를 다운로드할 수 있는 버튼 추가
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    st.download_button('Download Graph Image', buffer, file_name='graph.png', mime='image/png')


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
