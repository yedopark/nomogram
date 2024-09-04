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
   
Developer : Ye do Park, Chang BO Oh, Energy Safety Lab, Pukyong National University
Created on : 2024. 8. 20
Last updated : 2024. 8. 20

"""

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from PIL import Image

# 이미지 파일 경로 설정
image_directory = r'.'  # 현재 디렉토리
university_logo_path = f"{image_directory}/university_logo.png"
lab_logo_path = f"{image_directory}/lab_logo.png"

# 컬럼을 이용한 로고와 제목 배치
col1, col2 = st.columns([2.5, 6.5])

with col1:
    lab_logo = Image.open(lab_logo_path)
    st.image(lab_logo, use_column_width=True)

with col2:
    st.title("Calculation Program in Hydrogen Tank Explosion Overpressure and Impulse")
    st.write("This application calculates and visualizes data based on input pressure and volume.")

# 엑셀 파일 경로 설정
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
            return np.nan

        return round(float(result), 3)
    except Exception as e:
        st.error(f"Error in calculating impulse: {e}")
        return np.nan

# 사용자 입력 받기
pressure_input = st.number_input("압력을 입력하세요(MPa):", min_value=0.0, step=1.0)
volume_input = st.number_input("부피를 입력하세요(L):", min_value=0.0, step=1.0)

if st.button("계산 시작"):
    status_text.text("Loading Excel files...")
    
    # 엑셀 파일 읽기
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

    # Distance_2 (m) 배열로 저장 (첫 번째 열의 데이터)
    Distance_2 = df_first_sheet_impulse.index.values

    # C_data 계산
    C_data = calculate_c_data(df_first_sheet_impulse, pressure_input)
    C_data[C_data <= 0] = np.nan

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

    # 배열 길이 맞추기
    min_length_1 = min(len(A_data.dropna()), len(B_data_interpolated.dropna()), len(overpressure_values.dropna()))
    min_length_2 = min(len(Distance_2), len(C_data.dropna()), len(D_data), len(E_data), len(Impulse_data))

    # 첫 번째 시트용 데이터 생성
    output_df_1 = pd.DataFrame({
        'Distance_1 (m)': df_first_sheet_overpressure.index[:min_length_1],
        'A_data': A_data.dropna()[:min_length_1],
        'Overpressure (kPa)': overpressure_values.dropna()[:min_length_1],
    })

    # 두 번째 시트용 데이터 생성
    output_df_2 = pd.DataFrame({
        'Distance_2 (m)': Distance_2[:min_length_2],
        'C_data': C_data.dropna()[:min_length_2],
        'D_data': D_data[:min_length_2],
        'E_data': E_data[:min_length_2],
        'Impulse (kPa*s)': Impulse_data[:min_length_2],
    })

    # 엑셀 파일로 저장
    output_file_path = 'output_pressure_volume_data_with_impulse.xlsx'
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        output_df_1.to_excel(writer, sheet_name='Overpressure Data', index=False)
        output_df_2.to_excel(writer, sheet_name='Impulse Data', index=False)

    progress_bar.progress(100)
    status_text.text(f"Calculation complete. Results saved to {output_file_path}")

    # 엑셀 파일 다운로드 버튼
    with open(output_file_path, 'rb') as f:
        st.download_button('Download Excel File', f, file_name=output_file_path)

    # 그래프 생성
    st.write("### Graphs")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # y축 값이 0이 아닌 데이터 필터링
    filtered_output_df_1 = output_df_1[output_df_1['Overpressure (kPa)'] > 0]
    filtered_output_df_2 = output_df_2[output_df_2['Impulse (kPa*s)'] > 0]

    # 첫 번째 그래프: Overpressure (y축 로그 스케일)
    axs[0].plot(filtered_output_df_1['Distance_1 (m)'], filtered_output_df_1['Overpressure (kPa)'], marker='o', linestyle='-')
    axs[0].set_xscale('linear')
    axs[0].set_yscale('log')

    axs[0].set_xlabel('Distance (m)')
    axs[0].set_ylabel('Overpressure (kPa)')
    axs[0].set_title(f'{pressure_input}MPa, {volume_input}L ')

    # 두 번째 그래프: Impulse (x축을 Distance_2로 설정)
    axs[1].plot(filtered_output_df_2['Distance_2 (m)'], filtered_output_df_2['Impulse (kPa*s)'], marker='o', linestyle='-')
    axs[1].set_xscale('linear')
    axs[1].set_yscale('linear')

    axs[1].set_xlabel('Distance_2 (m)')
    axs[1].set_ylabel('Impulse (kPa*s)')
    axs[1].set_title('Impulse vs Distance_2 (m)')

    st.pyplot(fig)

    # 그래프 다운로드 버튼 추가
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    st.download_button('Download Graph Image', buffer, file_name='graph.png', mime='image/png')
