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

def calculate_total_efficiency(cmh, spmm, bkw, diameter):
    Q = cmh / 3600  # Convert CMH to m³/s
    A = get_outlet_area(diameter)
    V = Q / A  # Outlet velocity in m/s
    rho = 1.2  # Air density in kg/m³
    Pv = 0.5 * rho * V ** 2  # Velocity pressure in Pa
    P_static_Pa = spmm * 9.80665  # Static pressure in Pa
    P_total_Pa = P_static_Pa + Pv  # Total pressure in Pa
    eta_total = (Q * P_total_Pa) / (bkw * 1000) * 100  # Total efficiency in %
    return eta_total, V, Pv

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

# Debug state
st.write(f"Current page: {st.session_state.page}")

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
                eta_total, _, _ = calculate_total_efficiency(cmh_input, spmm, bkw, diameter)
                rpm = fan['DefaultOptions']['RPM']
                recommendations.append({
                    'title': title,
                    'eta_total': eta_total,
                    'rpm': rpm,
                    'stateff': stateff,
                    'bkw': bkw,
                    'spmm': spmm,
                    'diameter': diameter
                })
        
        if recommendations:
            # Sort and store in session state
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

    # Show model selection and View Details button only if recommendations exist
    if st.session_state.sorted_recommendations:
        selected_title = st.selectbox("Select a model", [rec['title'] for rec in st.session_state.sorted_recommendations])
        if st.button("View Details"):
            try:
                st.session_state.selected_model = next(rec for rec in st.session_state.sorted_recommendations if rec['title'] == selected_title)
                st.session_state.page = 'output'
                st.write("Page set to output")  # Debug statement
            except StopIteration:
                st.error("Selected model not found in recommendations.")

# Output page
if st.session_state.page == 'output':
    try:
        st.subheader(f"Selected Model: {st.session_state.selected_model['title']}")
        selected_fan = next(fan for fan in fan_data if fan['Title'] == st.session_state.selected_model['title'])
        cmh_input = st.session_state.cmh_input
        static_pr_input = st.session_state.static_pr_input
        diameter = st.session_state.selected_model['diameter']
        A = get_outlet_area(diameter)
        Q = cmh_input / 3600
        V = Q / A  # Outlet velocity
        rho = 1.2
        Pv_Pa = 0.5 * rho * V ** 2  # Velocity pressure in Pa
        Pv_MMWC = Pv_Pa / 9.80665  # Convert to MMWC
        spmm = st.session_state.selected_model['spmm']
        P_total_MMWC = static_pr_input + Pv_MMWC  # Total pressure = input SP + velocity pressure
        stateff = st.session_state.selected_model['stateff']
        eta_total = st.session_state.selected_model['eta_total']
        bkw = st.session_state.selected_model['bkw']
        rpm = st.session_state.selected_model['rpm']

        # Display calculated outputs with clarification
        st.write("### Calculated Outputs")
        st.write(f"**Required Static Pressure**: {static_pr_input:.2f} mmWC")
        st.write(f"**Fan's Delivered Static Pressure**: {spmm:.2f} mmWC")
        st.write(f"**Outlet Velocity**: {V:.2f} m/s")
        st.write(f"**Velocity Pressure**: {Pv_MMWC:.2f} mmWC")
        st.write(f"**Fan's Total Pressure**: {P_total_MMWC:.2f} mmWC")
        st.write(f"**Power (BKW)**: {bkw:.2f} kW")
        st.write(f"**RPM**: {rpm}")
        st.write(f"**Static Efficiency**: {stateff:.2f} %")
        st.write(f"**Total Efficiency**: {eta_total:.2f} %")

        # Plot performance curves
        data_table = selected_fan['DataTable']
        cmh_values = [point['CMH'] for point in data_table if point['CMH'] != ""]
        spmm_values = [point['SPMM'] for point in data_table if point['CMH'] != ""]
        stateff_values = [point['StatEff'] for point in data_table if point['CMH'] != ""]
        sysreg_values = [point['SysReg'] for point in data_table if point['CMH'] != ""]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot SPMM and SysReg on primary y-axis
        ax1.plot(cmh_values, spmm_values, label='Static Pressure (SPMM)', color='black')
        ax1.plot(cmh_values, sysreg_values, label='System Resistance (SysReg)', color='blue')
        ax1.set_xlabel('Capacity (CMH)')
        ax1.set_ylabel('Pressure (mmWC)')
        ax1.grid(True)

        # Create secondary y-axis for StatEff
        ax2 = ax1.twinx()
        ax2.plot(cmh_values, stateff_values, label='Static Efficiency (StatEff)', color='green')
        ax2.set_ylabel('Efficiency (%)')

        # Add vertical and horizontal lines
        ax1.axvline(x=cmh_input, color='red', linestyle='--', label=f'Input CMH = {cmh_input}')
        ax1.axhline(y=static_pr_input, color='red', linestyle='--', label=f'Input SP = {static_pr_input} mmWC')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Set title
        ax1.set_title('Fan Performance Curves')

        # Show plot
        st.pyplot(fig)

        if st.button("Back to Input"):
            st.session_state.page = 'input'
            st.session_state.sorted_recommendations = []  # Reset recommendations
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.page = 'input'
