import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit 설정
st.title("Nomogram Analysis")
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
            st.warning(f"NaN result encountered for volume: {volume}, e_data_value: {e_data_value}")

        # 배열 대신 단일 값으로 반환하도록 수정
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
        if np.isnan(e):
            st.warning(f"Skipping Impulse calculation for NaN E_data value: {e}")
            Impulse_data.append(np.nan)
        else:
            Impulse_data.append(calculate_impulse(df_fourth_sheet_impulse, volume_input, e))

    progress_bar.progress(90)
    status_text.text("Finalizing data...")

    # 배열들의 길이를 동일하게 맞춤
    min_length = min(len(A_data), len(B_data_interpolated), len(overpressure_values), len(C_data), len(D_data), len(E_data), len(Impulse_data))
    A_data = A_data.iloc[:min_length]
    B_data_interpolated = B_data_interpolated.iloc[:min_length]
    overpressure_values = overpressure_values.iloc[:min_length]
    C_data = C_data[:min_length]
    D_data = D_data[:min_length]
    E_data = E_data[:min_length]
    Impulse_data = Impulse_data[:min_length]

    # 결과를 엑셀 파일로 저장
    output_df = pd.DataFrame({
        'Disatance': df_first_sheet_overpressure.index[:min_length],
        'A_data': A_data,
        'B_data': B_data_interpolated,
        'Overpressure': overpressure_values,
        'C_data': C_data,
        'D_data': D_data,
        'E_data': E_data,
        'Impulse': Impulse_data
    })

    output_file_path = 'output_pressure_volume_data_with_impulse.xlsx'
    output_df.to_excel(output_file_path, index=False)

    progress_bar.progress(100)
    status_text.text(f"Calculation complete. Results saved to {output_file_path}")

    # 그래프 생성
    st.write("### Graphs")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 첫 번째 그래프: Overpressure (y축 로그 스케일)
    axs[0].plot(output_df['Disatance'], output_df['Overpressure'], marker='o', linestyle='-')
    axs[0].set_xscale('linear')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Disatance')
    axs[0].set_ylabel('Overpressure (log scale)')
    axs[0].set_title('Overpressure vs Disatance (Log Scale)')

    # 두 번째 그래프: Impulse
    axs[1].plot(output_df['Disatance'], output_df['Impulse'], marker='o', linestyle='-')
    axs[1].set_xscale('linear')
    axs[1].set_yscale('linear')
    axs[1].set_xlabel('Disatance')
    axs[1].set_ylabel('Impulse')
    axs[1].set_title('Impulse vs Disatance')

    st.pyplot(fig)
