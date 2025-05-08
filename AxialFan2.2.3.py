import streamlit as st
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import re

# Load fan data from external JSON file
with open('all.json', 'r') as f:
    fan_data = json.load(f)

# Helper functions
def get_diameter(title):
    match = re.search(r'(\d+)"', title)
    if match:
        diameter_inch = float(match.group(1))
        diameter_m = diameter_inch * 0.0254  # Convert inches to meters
        return diameter_m
    else:
        raise ValueError(f"Cannot extract diameter from title: {title}")

def get_outlet_area(diameter):
    return np.pi * (diameter / 2) ** 2

def interpolate_data(data_table, cmh):
    cmh_values = [point['CMH'] for point in data_table if point['CMH'] != ""]
    spmm_values = [point['SPMM'] for point in data_table if point['CMH'] != ""]
    bkw_values = [point['BKW'] for point in data_table if point['CMH'] != ""]
    stateff_values = [point['StatEff'] for point in data_table if point['CMH'] != ""]
    
    if not cmh_values or cmh < min(cmh_values) or cmh > max(cmh_values):
        return None  # Out of range or invalid data
    
    spmm_interp = interp1d(cmh_values, spmm_values, kind='linear')
    bkw_interp = interp1d(cmh_values, bkw_values, kind='linear')
    stateff_interp = interp1d(cmh_values, stateff_values, kind='linear')
    
    spmm = float(spmm_interp(cmh))
    bkw = float(bkw_interp(cmh))
    stateff = float(stateff_interp(cmh))
    
    return {'SPMM': spmm, 'BKW': bkw, 'StatEff': stateff}

def calculate_efficiencies(cmh_input, static_pr_input, P_total_MMWC, bkw):
    Q = cmh_input  # Input capacity in m³/hr
    P_static = static_pr_input  # Input static pressure in mmWC
    P_total = P_total_MMWC  # Total pressure in mmWC
    BKW = bkw  # Fan power in kW
    constant = 2.725e-3
    
    # Static efficiency using input values
    eta_static = (constant * Q * P_static) / (BKW * 1000) * 100
    
    # Total efficiency using total pressure
    eta_total = (constant * Q * P_total) / (BKW * 1000) * 100
    
    return eta_static, eta_total

# Streamlit app
st.title("Fan Selection Tool")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'sorted_recommendations' not in st.session_state:
    st.session_state.sorted_recommendations = []
if 'cmh_input' not in st.session_state:
    st.session_state.cmh_input = None
if 'static_pr_input' not in st.session_state:
    st.session_state.static_pr_input = None

# Input page
if st.session_state.page == 'input':
    st.subheader("Enter Fan Requirements")
    cmh_input = st.number_input("Enter CMH (Cubic Meters per Hour)", min_value=0.0, value=5000.0)
    static_pr_input = st.number_input("Enter Static Pressure (MMWC)", min_value=0.0, value=10.0)

    if st.button("Get Recommendations"):
        recommendations = []
        for fan in fan_data:
            title = fan['Title']
            try:
                diameter = get_diameter(title)
            except ValueError:
                continue  # Skip if diameter cannot be extracted
            data_table = fan['DataTable']
            interp_result = interpolate_data(data_table, cmh_input)
            if interp_result is None:
                continue  # Out of range or invalid data
            spmm = interp_result['SPMM']
            if spmm >= static_pr_input:
                bkw = interp_result['BKW']
                stateff = interp_result['StatEff']
                # Calculate velocity pressure for total pressure
                A = get_outlet_area(diameter)
                Q_m3s = cmh_input / 3600
                V = Q_m3s / A
                rho = 1.2
                Pv_Pa = 0.5 * rho * V ** 2
                Pv_MMWC = Pv_Pa / 9.80665
                P_total_MMWC = static_pr_input + Pv_MMWC
                # Calculate efficiencies
                eta_static, eta_total = calculate_efficiencies(cmh_input, static_pr_input, P_total_MMWC, bkw)
                rpm = fan['DefaultOptions']['RPM']
                recommendations.append({
                    'title': title,
                    'eta_total': eta_total,
                    'rpm': rpm,
                    'stateff': stateff,
                    'bkw': bkw,
                    'spmm': spmm,
                    'diameter': diameter,
                    'P_total_MMWC': P_total_MMWC
                })
        
        if recommendations:
            st.session_state.sorted_recommendations = sorted(recommendations, key=lambda x: (-x['eta_total'], x['rpm']))
            st.session_state.cmh_input = cmh_input
            st.session_state.static_pr_input = static_pr_input
            st.subheader("Recommended Fan Models")
            df = pd.DataFrame(st.session_state.sorted_recommendations)
            df_display = df[['title', 'eta_total', 'rpm', 'stateff', 'bkw']].rename(
                columns={'title': 'Model', 'eta_total': 'Total Efficiency (%)', 'rpm': 'RPM', 
                         'stateff': 'Static Efficiency (%)', 'bkw': 'Power (kW)'}
            )
            st.dataframe(df_display)
        else:
            st.warning("No models meet the specified criteria.")
            st.session_state.sorted_recommendations = []

    if st.session_state.sorted_recommendations:
        selected_title = st.selectbox("Select a model", [rec['title'] for rec in st.session_state.sorted_recommendations])
        if st.button("View Details"):
            try:
                st.session_state.selected_model = next(rec for rec in st.session_state.sorted_recommendations if rec['title'] == selected_title)
                st.session_state.page = 'output'
            except StopIteration:
                st.error("Selected model not found in recommendations.")

# Output page
if st.session_state.page == 'output':
    st.subheader(f"Selected Model: {st.session_state.selected_model['title']}")
    selected_fan = next(fan for fan in fan_data if fan['Title'] == st.session_state.selected_model['title'])
    cmh_input = st.session_state.cmh_input
    static_pr_input = st.session_state.static_pr_input
    diameter = st.session_state.selected_model['diameter']
    A = get_outlet_area(diameter)
    Q_m3s = cmh_input / 3600
    V = Q_m3s / A
    rho = 1.2
    Pv_Pa = 0.5 * rho * V ** 2
    Pv_MMWC = Pv_Pa / 9.80665
    spmm = st.session_state.selected_model['spmm']
    P_total_MMWC = static_pr_input + Pv_MMWC
    bkw = st.session_state.selected_model['bkw']
    rpm = st.session_state.selected_model['rpm']
    eta_static, eta_total = calculate_efficiencies(cmh_input, static_pr_input, P_total_MMWC, bkw)

    # Display calculated outputs
    st.write("### Calculated Outputs")
    st.write(f"**Required Static Pressure**: {static_pr_input:.2f} mmWC")
    st.write(f"**Fan's Delivered Static Pressure**: {spmm:.2f} mmWC")
    st.write(f"**Outlet Velocity**: {V:.2f} m/s")
    st.write(f"**Velocity Pressure**: {Pv_MMWC:.2f} mmWC")
    st.write(f"**Total Pressure**: {P_total_MMWC:.2f} mmWC")
    st.write(f"**Power (BKW)**: {bkw:.2f} kW")
    st.write(f"**RPM**: {rpm}")
    st.write(f"**Static Efficiency**: {eta_static:.2f} %")
    st.write(f"**Total Efficiency**: {eta_total:.2f} %")

    # Plot performance curves with subplots
    data_table = selected_fan['DataTable']
    cmh_values = [point['CMH'] for point in data_table if point['CMH'] != ""]
    spmm_values = [point['SPMM'] for point in data_table if point['CMH'] != ""]
    stateff_values = [point['StatEff'] for point in data_table if point['CMH'] != ""]
    sysreg_values = [point['SysReg'] for point in data_table if point['CMH'] != ""]
    bkw_values = [point['BKW'] for point in data_table if point['CMH'] != ""]

    # Create figure with two subplots
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top subplot: Pressure and Efficiency
    ax1.plot(cmh_values, spmm_values, label='Static Pressure (SPMM)', color='black')
    ax1.plot(cmh_values, sysreg_values, label='System Resistance (SysReg)', color='blue')
    ax1.set_ylabel('Pressure (mmWC)')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(cmh_values, stateff_values, label='Static Efficiency (StatEff)', color='green')
    ax2.set_ylabel('Efficiency (%)')

    ax1.axvline(x=cmh_input, color='red', linestyle='--', label=f'Input CMH = {cmh_input}')
    ax1.axhline(y=static_pr_input, color='red', linestyle='--', label=f'Input SP = {static_pr_input} mmWC')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title('Fan Performance: Pressure and Efficiency')

    # Bottom subplot: BKW
    ax3.plot(cmh_values, bkw_values, label='Power (BKW)', color='purple')
    ax3.set_xlabel('Capacity (CMH)')
    ax3.set_ylabel('Power (kW)')
    ax3.grid(True)
    ax3.axvline(x=cmh_input, color='red', linestyle='--', label=f'Input CMH = {cmh_input}')
    ax3.legend(loc='upper left')
    ax3.set_title('Fan Power Curve')

    plt.tight_layout()
    st.pyplot(fig)

    if st.button("Back to Input"):
        st.session_state.page = 'input'
        st.session_state.sorted_recommendations = []
