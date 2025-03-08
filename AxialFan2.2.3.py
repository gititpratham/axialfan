import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import math
from scipy.interpolate import interp1d

# Set page config
st.set_page_config(page_title="Fan Selection Tool", layout="wide")

# Function to load fan data from JSON file
def load_fan_data():
    # Look for JSON files in the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]
    
    if not json_files:
        st.error("No JSON files found in the current directory.")
        return None
    
    # Use the first JSON file found (you could add a file selector later)
    json_file = os.path.join(current_dir, json_files[0])
    
    with open(json_file, 'r') as f:
        return json.load(f)

# Function to find the best fan models for given requirements
def find_best_fans(fan_data, required_cmh, required_sp):
    recommended_fans = []
    
    for fan in fan_data:
        # Extract fan data
        fan_title = fan.get("Title", "Unknown")
        data_table = fan.get("DataTable", [])
        
        if not data_table:
            continue
        
        # Convert data table to DataFrame for easier analysis
        df = pd.DataFrame(data_table)
        
        # Check if we have enough data points
        if len(df) < 2:
            continue
            
        # Check if required columns exist
        required_cols = ["CMH", "SPMM", "StatEff", "BKW"]
        if not all(col in df.columns for col in required_cols):
            continue
            
        # Convert SPMM to Pa if needed
        if "SPMM" in df.columns:
            df["SP_Pa"] = df["SPMM"] * 9.81  # Convert mm water column to Pa
        
        # Try to interpolate performance at the required point
        try:
            # Sort by CMH for proper interpolation
            df = df.sort_values("CMH")
            
            # Create interpolation functions
            if len(df) >= 2:
                f_sp = interp1d(df["CMH"], df["SP_Pa"], bounds_error=False, fill_value="extrapolate")
                f_eff = interp1d(df["CMH"], df["StatEff"], bounds_error=False, fill_value="extrapolate")
                f_power = interp1d(df["CMH"], df["BKW"], bounds_error=False, fill_value="extrapolate")
                
                # Get performance at required flow
                interp_sp = float(f_sp(required_cmh))
                interp_eff = float(f_eff(required_cmh))
                interp_power = float(f_power(required_cmh))
                
                # Get RPM from fan data
                rpm = fan.get("DefaultOptions", {}).get("RPM", 0)
                
                # Calculate performance difference
                sp_diff = abs(interp_sp - required_sp)
                
                # Add to recommended list if within reasonable range 
                # (20% difference in static pressure)
                if sp_diff / required_sp <= 0.2:
                    recommended_fans.append({
                        "fan_title": fan_title,
                        "fan_data": fan,
                        "interp_eff": interp_eff,
                        "interp_sp": interp_sp,
                        "interp_power": interp_power,
                        "rpm": rpm,
                        "sp_diff": sp_diff
                    })
        except Exception as e:
            # Skip this fan if interpolation fails
            continue
    
    # Sort by efficiency (descending) and then by RPM (ascending)
    if recommended_fans:
        recommended_fans.sort(key=lambda x: (-x["interp_eff"], x["rpm"]))
    
    return recommended_fans

# Function to calculate additional fan parameters
def calculate_fan_parameters(fan_data, cmh, sp_pa):
    # Extract fan data 
    data_table = fan_data.get("DataTable", [])
    df = pd.DataFrame(data_table)
    
    # Convert mm water column to Pa if needed
    if "SPMM" in df.columns:
        df["SP_Pa"] = df["SPMM"] * 9.81
    
    # Sort by CMH for proper interpolation
    df = df.sort_values("CMH")
    
    # Create interpolation functions
    f_sp = interp1d(df["CMH"], df["SP_Pa"], bounds_error=False, fill_value="extrapolate")
    f_eff = interp1d(df["CMH"], df["StatEff"], bounds_error=False, fill_value="extrapolate")
    f_power = interp1d(df["CMH"], df["BKW"], bounds_error=False, fill_value="extrapolate")
    
    # Calculate parameters
    # Assuming standard air density of 1.2 kg/m³
    air_density = 1.2
    
    # Convert CMH to m³/s
    flow_m3s = cmh / 3600
    
    # Static pressure in Pa
    static_pressure = sp_pa
    
    # Calculate cross-sectional area (estimated based on flow and typical velocity)
    # For a more accurate calculation, we'd need the actual fan dimensions
    estimated_velocity = 10  # typical velocity in m/s
    estimated_area = flow_m3s / estimated_velocity
    
    # Calculate more accurate velocity using the estimated area
    outlet_velocity = flow_m3s / estimated_area
    
    # Velocity pressure
    velocity_pressure = 0.5 * air_density * (outlet_velocity ** 2)
    
    # Total pressure
    total_pressure = static_pressure + velocity_pressure
    
    # Static efficiency from interpolation
    static_efficiency = float(f_eff(cmh))
    
    # Total efficiency (estimated)
    power_input = float(f_power(cmh))
    if power_input > 0:
        total_efficiency = (flow_m3s * total_pressure) / (power_input * 1000) * 100
    else:
        total_efficiency = 0
    
    results = {
        "Outlet Velocity (m/s)": round(outlet_velocity, 2),
        "Velocity Pressure (Pa)": round(velocity_pressure, 2),
        "Total Pressure (Pa)": round(total_pressure, 2),
        "Static Efficiency (%)": round(static_efficiency, 2),
        "Total Efficiency (%)": round(total_efficiency, 2),
        "Power Input (kW)": round(power_input, 3)
    }
    
    return results

# Function to generate performance curves
def generate_performance_graph(fan_data, required_cmh, required_sp):
    # Extract fan data
    data_table = fan_data.get("DataTable", [])
    df = pd.DataFrame(data_table)
    
    # Convert mm water column to Pa if needed
    if "SPMM" in df.columns:
        df["SP_Pa"] = df["SPMM"] * 9.81
    
    # Sort by CMH for proper graphing
    df = df.sort_values("CMH")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Static Pressure vs. Flow
    ax1.plot(df["CMH"], df["SP_Pa"], 'k-', linewidth=2)
    ax1.set_xlabel('Volume Flow (m³/h)')
    ax1.set_ylabel('Static Pressure (Pa)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add red lines for the required point
    ax1.axvline(x=required_cmh, color='red', linestyle='-')
    ax1.axhline(y=required_sp, color='red', linestyle='-')
    
    # Plot 2: Power vs. Flow
    ax2.plot(df["CMH"], df["BKW"], 'k-', linewidth=2)
    ax2.set_xlabel('Volume Flow (m³/h)')
    ax2.set_ylabel('Power (kW)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add red line for the required flow
    ax2.axvline(x=required_cmh, color='red', linestyle='-')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Main application
def main():
    st.title("Fan Selection and Analysis Tool")
    
    # Load fan data
    fan_data = load_fan_data()
    if not fan_data:
        st.stop()
    
    # Application state
    if 'page' not in st.session_state:
        st.session_state.page = 'input'
    if 'selected_fan' not in st.session_state:
        st.session_state.selected_fan = None
    if 'required_cmh' not in st.session_state:
        st.session_state.required_cmh = 0
    if 'required_sp' not in st.session_state:
        st.session_state.required_sp = 0
    
    # Input page
    if st.session_state.page == 'input':
        st.header("Enter Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            required_cmh = st.number_input("Volume Flow (CMH)", min_value=0.0, value=1000.0, step=100.0)
            
        with col2:
            pressure_unit = st.selectbox("Pressure Unit", ["Pa", "mm H₂O"])
            if pressure_unit == "Pa":
                required_sp_input = st.number_input("Static Pressure", min_value=0.0, value=100.0, step=10.0)
                required_sp = required_sp_input  # Already in Pa
            else:
                required_sp_input = st.number_input("Static Pressure", min_value=0.0, value=10.0, step=1.0)
                required_sp = required_sp_input * 9.81  # Convert to Pa
        
        if st.button("Find Best Fans"):
            st.session_state.required_cmh = required_cmh
            st.session_state.required_sp = required_sp
            st.session_state.page = 'recommendations'
    
    # Recommendations page
    elif st.session_state.page == 'recommendations':
        st.header("Recommended Fans")
        
        # Display requirements
        st.write(f"Volume Flow: {st.session_state.required_cmh} m³/h")
        if pressure_unit == "Pa":
            st.write(f"Static Pressure: {st.session_state.required_sp} Pa")
        else:
            st.write(f"Static Pressure: {st.session_state.required_sp/9.81:.2f} mm H₂O ({st.session_state.required_sp:.2f} Pa)")
        
        # Find best fans
        recommended_fans = find_best_fans(fan_data, st.session_state.required_cmh, st.session_state.required_sp)
        
        if not recommended_fans:
            st.warning("No suitable fans found for the given requirements.")
            if st.button("Back to Input"):
                st.session_state.page = 'input'
            st.stop()
        
        # Display recommendations
        st.write(f"Found {len(recommended_fans)} suitable fan models:")
        
        for i, fan in enumerate(recommended_fans):
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
            
            with col1:
                st.write(f"**{fan['fan_title']}**")
            with col2:
                st.write(f"Efficiency: {fan['interp_eff']:.2f}%")
            with col3:
                st.write(f"Static Pressure: {fan['interp_sp']:.2f} Pa")
            with col4:
                st.write(f"RPM: {fan['rpm']}")
            with col5:
                if st.button("Select", key=f"select_{i}"):
                    st.session_state.selected_fan = fan
                    st.session_state.page = 'fan_details'
        
        if st.button("Back"):
            st.session_state.page = 'input'
    
    # Fan details page
    elif st.session_state.page == 'fan_details':
        if not st.session_state.selected_fan:
            st.error("No fan selected. Please go back and select a fan.")
            if st.button("Back to Recommendations"):
                st.session_state.page = 'recommendations'
            st.stop()
        
        fan = st.session_state.selected_fan
        st.header(f"Fan Model: {fan['fan_title']}")
        
        # Display requirements
        st.subheader("Requirements")
        st.write(f"Volume Flow: {st.session_state.required_cmh} m³/h")
        st.write(f"Static Pressure: {st.session_state.required_sp:.2f} Pa")
        
        # Calculate additional parameters
        params = calculate_fan_parameters(fan['fan_data'], st.session_state.required_cmh, st.session_state.required_sp)
        
        # Display parameters
        st.subheader("Fan Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            for key, value in list(params.items())[:3]:
                st.metric(key, value)
        
        with col2:
            for key, value in list(params.items())[3:]:
                st.metric(key, value)
        
        # Generate and display performance graph
        st.subheader("Performance Curves")
        fig = generate_performance_graph(fan['fan_data'], st.session_state.required_cmh, st.session_state.required_sp)
        st.pyplot(fig)
        
        # Fan details
        st.subheader("Fan Specifications")
        default_options = fan['fan_data'].get("DefaultOptions", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**RPM:** {default_options.get('RPM', 'N/A')}")
            st.write(f"**Power Rating:** {default_options.get('KW', 'N/A')} kW")
        
        with col2:
            st.write(f"**Default CMH:** {default_options.get('CMH', 'N/A')}")
            st.write(f"**Default Pressure:** {default_options.get('MMWC', 'N/A')} mm H₂O")
        
        if st.button("Back to Recommendations"):
            st.session_state.page = 'recommendations'
        
        if st.button("New Selection"):
            st.session_state.page = 'input'

if __name__ == "__main__":
    main()