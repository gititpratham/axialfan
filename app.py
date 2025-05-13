import streamlit as st
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import re
import base64
from io import BytesIO
from PIL import Image
import matplotlib.dates as mdates
from datetime import datetime
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib
matplotlib.use('Agg')

# Standard motor sizes (ascending by kW)
STANDARD_MOTORS = [
    (0.37, 0.5),  (0.55, 0.75), (0.75, 1),   (1.1, 1.5),    (1.5, 2),     
    (2.2, 3),    (3.7, 5),     (5.5, 7.5),    (7.5, 10),    (9.3, 12.5), 
    (11, 15),     (15, 20),    (18.5, 25),   (22, 30),    (30, 40),     
    (37, 50),    (45, 60),     (55, 75),    (75, 100),    (90, 120),    
    (110, 150),   (125, 170),  (132, 180),   (160, 215),    (180, 240),   
    (200, 270),  (225, 300),   (250, 335),    (275, 370),   (315, 425)
]

def pick_motor(pc_kw):
    """
    Return the smallest standard motor ≥ 120% of pc_kw.
    
    Args:
        pc_kw (float): Power consumption in kilowatts
    
    Returns:
        str: Formatted string with kW and HP of preferred motor
    """
    target = pc_kw * 1.2
    for kw, hp in STANDARD_MOTORS:
        if kw >= target:
            return f"{kw} kW/{hp} HP"
    
    # Fallback to largest motor if none big enough
    kw, hp = STANDARD_MOTORS[-1]
    return f"{kw} kW/{hp} HP"

# Set page config
st.set_page_config(
    page_title="MaximAir - Fan Selection Tool",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the MaximAir color theme
PRIMARY_COLOR = "#00D9A9"  # Green from logo
SECONDARY_COLOR = "#808080"  # Gray from logo
LIGHT_GRAY = "#F0F2F6"
DARK_GRAY = "#555555"

# Custom CSS
st.markdown(f"""
<style>
    .reportview-container .main .block-container{{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {DARK_GRAY};
        font-family: 'Arial', sans-serif;
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: {DARK_GRAY};
    }}
    .stNumberInput>div>div>input {{
        border-radius: 5px;
    }}
    .stSelectbox>div>div>div {{
        border-radius: 5px;
    }}
    .css-1d391kg {{
        background-color: {LIGHT_GRAY};
    }}
    footer {{
        visibility: hidden;
    }}
    .stProgress {{
        background-color: {LIGHT_GRAY};
    }}
    .stProgress > div > div {{
        background-color: {PRIMARY_COLOR};
    }}
    .css-184tjsw p {{
        font-size: 1.1rem;
    }}
</style>
""", unsafe_allow_html=True)

# Load fan data from external JSON file
@st.cache_data
def load_fan_data():
    with open('all.json', 'r') as f:
        return json.load(f)

# Create a function to convert units
def convert_flow_rate(value, from_unit, to_unit):
    # Convert to m³/h (CMH) from any unit
    if from_unit == "CMH":
        cmh = value
    elif from_unit == "CFM":
        cmh = value * 1.699  # CFM to CMH
    elif from_unit == "m³/s":
        cmh = value * 3600  # m³/s to CMH
    elif from_unit == "L/s":
        cmh = value * 3.6  # L/s to CMH
    
    # Convert from CMH to target unit
    if to_unit == "CMH":
        return cmh
    elif to_unit == "CFM":
        return cmh / 1.699  # CMH to CFM
    elif to_unit == "m³/s":
        return cmh / 3600  # CMH to m³/s
    elif to_unit == "L/s":
        return cmh / 3.6  # CMH to L/s

def convert_pressure(value, from_unit, to_unit):
    # Convert to mmWC from any unit
    if from_unit == "mmWC":
        mmwc = value
    elif from_unit == "Pa":
        mmwc = value / 9.80665  # Pa to mmWC
    elif from_unit == "inWC":
        mmwc = value * 25.4  # inWC to mmWC
    elif from_unit == "bar":
        mmwc = value * 10197.16  # bar to mmWC
    elif from_unit == "kPa":
        mmwc = value * 101.9716  # kPa to mmWC
    
    # Convert from mmWC to target unit
    if to_unit == "mmWC":
        return mmwc
    elif to_unit == "Pa":
        return mmwc * 9.80665  # mmWC to Pa
    elif to_unit == "inWC":
        return mmwc / 25.4  # mmWC to inWC
    elif to_unit == "bar":
        return mmwc / 10197.16  # mmWC to bar
    elif to_unit == "kPa":
        return mmwc / 101.9716  # mmWC to kPa

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

def encode_img_to_b64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Function to create a professionally formatted PDF report
def generate_report(project_name, company_info, input_data, output_data, performance_plot):
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add custom styles
    styles.add(ParagraphStyle(
        name='CustomTitle',
        fontSize=18,
        leading=22,
        alignment=1,
        spaceAfter=10
    ))

    
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=1,
        spaceAfter=6
    ))
    
    styles.add(ParagraphStyle(
        name='CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    ))
    
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor("#00D9A9"),
        spaceAfter=6,
        spaceBefore=12
    ))
    
    # Add logo and company name
    logo_data = io.BytesIO()
    company_info['logo'].save(logo_data, format='PNG')
    logo_data.seek(0)
    
    # Create a header with logo and company info
    header_tbl_data = [
        [RLImage(logo_data, width=2*inch, height=1*inch), 
         Paragraph(f"<b>MaximAir</b><br/>{company_info['tagline']}", styles["Normal"])]
    ]
    
    header_tbl = Table(header_tbl_data, colWidths=[2.5*inch, 4*inch])
    header_tbl.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    
    elements.append(header_tbl)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add title
    elements.append(Paragraph(f"Fan Selection Report", styles["Title"]))
    elements.append(Paragraph(f"Project: {project_name}", styles["Subtitle"]))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles["Normal"]))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add Input Data Section
    elements.append(Paragraph("Input Parameters", styles["SectionTitle"]))
    input_data_tbl = [
                ["Parameter", "Value", "Unit"],
                ["Flow Rate", f"{input_data['flow_value']:.2f}", input_data['flow_unit']],
                ["Static Pressure", f"{input_data['pressure_value']:.2f}", input_data['pressure_unit']],
                ["Preferred Motor", input_data['preferred_motor'], "kW/HP"]  # New row
            ]
    
    tbl = Table(input_data_tbl, colWidths=[2.2*inch, 1.5*inch, 1.5*inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#00D9A9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.25*inch))
    
    # Selected Fan Section
    elements.append(Paragraph("Selected Fan", styles["SectionTitle"]))
    selected_fan_tbl = [
        ["Model", output_data['title']],
        ["Diameter", f"{output_data['diameter']*100:.1f} cm ({output_data['diameter']*100/2.54:.1f} in)"],
        ["RPM", f"{output_data['rpm']:.0f}"],
    ]
    
    tbl = Table(selected_fan_tbl, colWidths=[2.5*inch, 4*inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F0F2F6")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.25*inch))
    
    # Performance Results Section
    elements.append(Paragraph("Performance Results", styles["SectionTitle"]))
    performance_tbl = [
        ["Parameter", "Value", "Unit"],
        ["Static Pressure (Required)", f"{output_data['static_pr_input']:.2f}", "mmWC"],
        ["Static Pressure (Delivered)", f"{output_data['spmm']:.2f}", "mmWC"],
        ["Outlet Velocity", f"{output_data['V']:.2f}", "m/s"],
        ["Velocity Pressure", f"{output_data['Pv_MMWC']:.2f}", "mmWC"],
        ["Total Pressure", f"{output_data['P_total_MMWC']:.2f}", "mmWC"],
        ["Power Consumption", f"{output_data['bkw']:.2f}", "kW"],
        ["Static Efficiency", f"{output_data['eta_static']:.2f}", "%"],
        ["Total Efficiency", f"{output_data['eta_total']:.2f}", "%"],
    ]
    
    tbl = Table(performance_tbl, colWidths=[2.2*inch, 1.5*inch, 1.5*inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#00D9A9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add Performance Curve
    elements.append(Paragraph("Performance Curves", styles["SectionTitle"]))
    if performance_plot:
        img_data = BytesIO()
        performance_plot.savefig(img_data, format='png', dpi=300, bbox_inches='tight')
        img_data.seek(0)
        elements.append(RLImage(img_data, width=6.5*inch, height=5*inch))
    
    # Add footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = "MaximAir - Environment As Per Requirement"
    elements.append(Paragraph(f"<i>{footer_text}</i>", styles["Normal"]))
    elements.append(Paragraph(f"<i>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</i>", styles["Normal"]))
    
    # Build the PDF
    document.build(elements)
    buffer.seek(0)
    return buffer

def main():
    fan_data = load_fan_data()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'input'
    if 'sorted_recommendations' not in st.session_state:
        st.session_state.sorted_recommendations = []
    if 'cmh_input' not in st.session_state:
        st.session_state.cmh_input = None
    if 'static_pr_input' not in st.session_state:
        st.session_state.static_pr_input = None
    if 'flow_unit' not in st.session_state:
        st.session_state.flow_unit = "CMH"
    if 'pressure_unit' not in st.session_state:
        st.session_state.pressure_unit = "mmWC"
    if 'project_name' not in st.session_state:
        st.session_state.project_name = ""
        
    # Try to load logo
    try:
        logo = Image.open("logo.png")
        st.sidebar.image(logo, width=200)
    except:
        # If logo file is not available, create a placeholder
        st.sidebar.markdown(f"""
        <div style="background-color:{PRIMARY_COLOR}; padding:10px; border-radius:5px; text-align:center;">
            <h2 style="color:white;">MaximAir</h2>
            <p style="color:white;">Environment As Per Requirement</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Fan Selection Tool")
    
    # Input page
    if st.session_state.page == 'input':
        # Create two columns for the title and a placeholder for the logo
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("MaximAir Fan Selection Tool")
        
        st.markdown("<p style='font-size: 1.2em;'>Enter your project details and fan requirements</p>", unsafe_allow_html=True)
        
        # Project information
        with st.container():
            st.subheader("Project Information")
            project_name = st.text_input("Project Name", value=st.session_state.project_name)
            st.session_state.project_name = project_name
        
        # Fan requirements with unit selection
        with st.container():
            st.subheader("Fan Requirements")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Flow rate with unit selection
                st.markdown("#### Flow Rate")
                flow_col1, flow_col2 = st.columns([3, 1])
                with flow_col1:
                    flow_value = st.number_input("Enter flow rate", 
                                               min_value=0.0, 
                                               value=5000.0 if st.session_state.flow_unit == "CMH" else 
                                                     convert_flow_rate(5000.0, "CMH", st.session_state.flow_unit),
                                               step=10.0)
                with flow_col2:
                    flow_unit = st.selectbox("Unit", 
                                         ["CMH", "CFM", "m³/s", "L/s"],
                                         index=["CMH", "CFM", "m³/s", "L/s"].index(st.session_state.flow_unit))
                    if flow_unit != st.session_state.flow_unit:
                        st.session_state.flow_unit = flow_unit
            
            with col2:
                # Pressure with unit selection
                st.markdown("#### Static Pressure")
                pressure_col1, pressure_col2 = st.columns([3, 1])
                with pressure_col1:
                    pressure_value = st.number_input("Enter static pressure", 
                                                  min_value=0.0, 
                                                  value=10.0 if st.session_state.pressure_unit == "mmWC" else 
                                                        convert_pressure(10.0, "mmWC", st.session_state.pressure_unit),
                                                  step=1.0)
                with pressure_col2:
                    pressure_unit = st.selectbox("Unit", 
                                            ["mmWC", "Pa", "inWC", "bar", "kPa"],
                                            index=["mmWC", "Pa", "inWC", "bar", "kPa"].index(st.session_state.pressure_unit))
                    if pressure_unit != st.session_state.pressure_unit:
                        st.session_state.pressure_unit = pressure_unit
        
        # Convert input values to standard units (CMH and mmWC)
        cmh_input = convert_flow_rate(flow_value, flow_unit, "CMH")
        static_pr_input = convert_pressure(pressure_value, pressure_unit, "mmWC")
        
        # Button to calculate
        col1, col2 = st.columns([1, 5])
        with col1:
            calculate_btn = st.button("Find Fans", use_container_width=True)
        
        if calculate_btn:
            with st.spinner("Calculating optimal fan models..."):
                # Store standard units for calculations
                st.session_state.cmh_input = cmh_input
                st.session_state.static_pr_input = static_pr_input
                
                # Find recommendations
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
                    
                    # Display recommendations in a nice table
                    st.subheader("Recommended Fan Models")
                    
                    # Style the dataframe with CSS
                    df = pd.DataFrame(st.session_state.sorted_recommendations)
                    df_display = df[['title', 'eta_total', 'rpm', 'stateff', 'bkw']].rename(
                        columns={
                            'title': 'Model', 
                            'eta_total': 'Total Efficiency (%)', 
                            'rpm': 'RPM', 
                            'stateff': 'Static Efficiency (%)', 
                            'bkw': 'Power (kW)'
                        }
                    )
                    
                    # Format the columns
                    df_display['Total Efficiency (%)'] = df_display['Total Efficiency (%)'].map('{:.2f}'.format)
                    df_display['Static Efficiency (%)'] = df_display['Static Efficiency (%)'].map('{:.2f}'.format)
                    df_display['Power (kW)'] = df_display['Power (kW)'].map('{:.2f}'.format)
                    
                    # Display the table with improved styling
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Model": st.column_config.TextColumn("Model", width="medium"),
                            "Total Efficiency (%)": st.column_config.NumberColumn(
                                "Total Efficiency (%)",
                                format="%.2f%%",
                                width="small"
                            ),
                            "RPM": st.column_config.NumberColumn(
                                "RPM",
                                format="%d",
                                width="small"
                            ),
                            "Static Efficiency (%)": st.column_config.NumberColumn(
                                "Static Efficiency (%)",
                                format="%.2f%%",
                                width="small"
                            ),
                            "Power (kW)": st.column_config.NumberColumn(
                                "Power (kW)",
                                format="%.2f",
                                width="small"
                            )
                        }
                    )
                else:
                    st.error("No models meet the specified criteria. Please adjust your requirements.")
                    st.session_state.sorted_recommendations = []

        if st.session_state.sorted_recommendations:
            st.markdown("### Select a model for detailed analysis")
            selected_title = st.selectbox(
                "Choose a fan model",
                [rec['title'] for rec in st.session_state.sorted_recommendations],
                index=0
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("View Details", use_container_width=True):
                    try:
                        st.session_state.selected_model = next(rec for rec in st.session_state.sorted_recommendations if rec['title'] == selected_title)
                        st.session_state.page = 'output'
                        st.rerun()
                    except StopIteration:
                        st.error("Selected model not found in recommendations.")

    # Output page
    if st.session_state.page == 'output':
        # Create two columns for the title
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("Fan Performance Analysis")
        
        # Get the selected model data
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
        
        # Store outputs in a dictionary for the report
        output_data = {
            'title': st.session_state.selected_model['title'],
            'diameter': diameter,
            'static_pr_input': static_pr_input,
            'spmm': spmm,
            'V': V,
            'Pv_MMWC': Pv_MMWC,
            'P_total_MMWC': P_total_MMWC,
            'bkw': bkw,
            'rpm': rpm,
            'eta_static': eta_static,
            'eta_total': eta_total
        }
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📊 Performance Data", "📈 Performance Curves", "📄 Generate Report"])
        
        with tab1:
            # Display model details in a nice card
            st.subheader(f"Selected Model: {st.session_state.selected_model['title']}")
            
            # Create two columns for model details and performance
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Specifications")
                specs_df = pd.DataFrame({
                    'Parameter': ['Diameter', 'RPM'],
                    'Value': [f"{diameter*100:.1f} cm ({diameter*100/2.54:.1f} in)", f"{rpm:.0f}"]
                })
                
                st.dataframe(
                    specs_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.markdown("#### Project Requirements")
                req_df = pd.DataFrame({
                    'Parameter': ['Flow Rate', 'Static Pressure'],
                    'Value': [
                        f"{convert_flow_rate(cmh_input, 'CMH', st.session_state.flow_unit):.2f} {st.session_state.flow_unit}",
                        f"{convert_pressure(static_pr_input, 'mmWC', st.session_state.pressure_unit):.2f} {st.session_state.pressure_unit}"
                    ]
                })
                
                st.dataframe(
                    req_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Display calculated performance data
            st.markdown("#### Performance Results")
            
            # Create a DataFrame for the performance results
            perf_data = {
            'Parameter': [
                'Static Pressure (Required)', 
                'Static Pressure (Delivered)', 
                'Outlet Velocity', 
                'Velocity Pressure', 
                'Total Pressure', 
                'Power Consumption', 
                'Preferred Motor',  # New column
                'Static Efficiency', 
                'Total Efficiency'
            ],
            'Value': [
                f"{static_pr_input:.2f} mmWC",
                f"{spmm:.2f} mmWC",
                f"{V:.2f} m/s",
                f"{Pv_MMWC:.2f} mmWC",
                f"{P_total_MMWC:.2f} mmWC",
                f"{bkw:.2f} kW",
                pick_motor(bkw),  # Add preferred motor selection
                f"{eta_static:.2f}%",
                f"{eta_total:.2f}%"
            ]
        }
        
        perf_df = pd.DataFrame(perf_data)
        
            
            # Add color-coded metric display for key parameters
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric(
                "Total Efficiency", 
                f"{eta_total:.2f}%", 
                f"{eta_total - eta_static:.2f}%" if eta_total > eta_static else f"{eta_total - eta_static:.2f}%")
        with metric_cols[1]:
            st.metric(
                "Static Pressure", 
                f"{spmm:.2f} mmWC", 
                f"{spmm - static_pr_input:.2f}" if spmm > static_pr_input else f"{spmm - static_pr_input:.2f}")
        with metric_cols[2]:
            st.metric("Power", f"{bkw:.2f} kW")
        with metric_cols[3]:
            st.metric("RPM", f"{rpm:.0f}")
            
            # Show detailed performance table
        st.dataframe(
            perf_df,
            hide_index=True,
            use_container_width=True
        )
            
        with tab2:
            st.subheader("Performance Curves")
            
            # Get data for plotting
            data_table = selected_fan['DataTable']
            cmh_values = [point['CMH'] for point in data_table if point['CMH'] != ""]
            spmm_values = [point['SPMM'] for point in data_table if point['CMH'] != ""]
            stateff_values = [point['StatEff'] for point in data_table if point['CMH'] != ""]
            sysreg_values = [point['SysReg'] for point in data_table if point['CMH'] != ""]
            bkw_values = [point['BKW'] for point in data_table if point['CMH'] != ""]
            
            # Use Plotly for more attractive, interactive plots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Pressure & Efficiency Curves", "Power Curve"),
                vertical_spacing=0.15,
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )
            
            # Add traces for pressure plot
            fig.add_trace(
                go.Scatter(
                    x=cmh_values, y=spmm_values,
                    name="Static Pressure",
                    line=dict(color=DARK_GRAY, width=3),
                    hovertemplate="Flow: %{x:.0f} CMH<br>Static Pressure: %{y:.2f} mmWC"
                ),
                row=1, col=1, secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=cmh_values, y=sysreg_values,
                    name="System Resistance",
                    line=dict(color="blue", width=2, dash="dash"),
                    hovertemplate="Flow: %{x:.0f} CMH<br>System Resistance: %{y:.2f} mmWC"
                ),
                row=1, col=1, secondary_y=False
            )
            
            # Add efficiency trace (secondary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=cmh_values, y=stateff_values,
                    name="Static Efficiency",
                    line=dict(color=PRIMARY_COLOR, width=2),
                    hovertemplate="Flow: %{x:.0f} CMH<br>Efficiency: %{y:.2f}%"
                ),
                row=1, col=1, secondary_y=True
            )
            
            # Add power curve (bottom plot)
            fig.add_trace(
                go.Scatter(
                    x=cmh_values, y=bkw_values,
                    name="Power",
                    line=dict(color="purple", width=3),
                    hovertemplate="Flow: %{x:.0f} CMH<br>Power: %{y:.2f} kW",
                    fill='tozeroy',
                    fillcolor='rgba(128, 0, 128, 0.1)'
                ),
                row=2, col=1
            )
            
            # Add vertical line for the selected flow rate
            fig.add_vline(
                x=cmh_input, line_width=2, line_dash="dot", line_color="red",
                annotation_text=f"Selected: {cmh_input:.0f} CMH",
                annotation_position="top right"
            )
            
            # Add horizontal line for required static pressure
            fig.add_hline(
                y=static_pr_input, line_width=2, line_dash="dot", line_color="red",
                row=1, col=1,
                annotation_text=f"Required: {static_pr_input:.2f} mmWC",
                annotation_position="top left"
            )
            
            # Update layout
            fig.update_layout(
                title=f"Fan Performance: {st.session_state.selected_model['title']}",
                height=700,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                font=dict(family="Arial"),
                hovermode="x unified",
                margin=dict(l=20, r=20, t=60, b=30),
                plot_bgcolor='white'
            )
            
            # Update y-axes titles
            fig.update_yaxes(title_text="Pressure (mmWC)", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
            
            # Update x-axes titles
            fig.update_xaxes(title_text="", row=1, col=1)
            fig.update_xaxes(title_text="Flow Rate (CMH)", row=2, col=1)
            
            # Add grid lines for better readability
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Also create a matplotlib version for the PDF report
            performance_plot_fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Top subplot: Pressure and Efficiency
            ax1.plot(cmh_values, spmm_values, label='Static Pressure (SPMM)', color='black', linewidth=2)
            ax1.plot(cmh_values, sysreg_values, label='System Resistance (SysReg)', color='blue', linewidth=2, linestyle='--')
            ax1.set_ylabel('Pressure (mmWC)')
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            ax2.plot(cmh_values, stateff_values, label='Static Efficiency (StatEff)', color=PRIMARY_COLOR, linewidth=2)
            ax2.set_ylabel('Efficiency (%)')
            
            ax1.axvline(x=cmh_input, color='red', linestyle='--', linewidth=1.5, label=f'Input CMH = {cmh_input}')
            ax1.axhline(y=static_pr_input, color='red', linestyle='--', linewidth=1.5, label=f'Input SP = {static_pr_input} mmWC')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax1.set_title(f'Fan Performance: {st.session_state.selected_model["title"]}')
            
            # Bottom subplot: BKW
            ax3.plot(cmh_values, bkw_values, label='Power (BKW)', color='purple', linewidth=2)
            ax3.fill_between(cmh_values, bkw_values, alpha=0.1, color='purple')
            ax3.set_xlabel('Capacity (CMH)')
            ax3.set_ylabel('Power (kW)')
            ax3.grid(True, alpha=0.3)
            ax3.axvline(x=cmh_input, color='red', linestyle='--', linewidth=1.5, label=f'Input CMH = {cmh_input}')
            ax3.legend(loc='upper left')
            
            plt.tight_layout()
            
        with tab3:
            st.subheader("Generate Professional Report")
            
            # Allow user to provide or update project name
            project_name = st.text_input("Project Name", value=st.session_state.project_name)
            st.session_state.project_name = project_name
            
            # Prepare company info
            try:
                logo = Image.open("logo.png")
            except:
                # Create a dummy logo with the company colors
                logo = Image.new('RGB', (200, 100), color=PRIMARY_COLOR)
            
            company_info = {
                'logo': logo,
                'tagline': "Environment As Per Requirement"
            }
            
            # Prepare input data for report
            input_data = {
            'flow_value': convert_flow_rate(cmh_input, 'CMH', st.session_state.flow_unit),
            'flow_unit': st.session_state.flow_unit,
            'pressure_value': convert_pressure(static_pr_input, 'mmWC', st.session_state.pressure_unit),
            'pressure_unit': st.session_state.pressure_unit,
            'preferred_motor': pick_motor(bkw)  # Add preferred motor to input data
            }
            
            # Generate PDF report when requested
            if st.button("Generate PDF Report"):
                with st.spinner("Generating professional report..."):
                    pdf_buffer = generate_report(project_name, company_info, input_data, output_data, performance_plot_fig)
                    
                    # Provide download button for the generated PDF
                    st.success("Report generated successfully!")
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"MaximAir_Fan_Report_{project_name.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        # Back button
        if st.button("← Back to Fan Selection"):
            st.session_state.page = 'input'
            st.rerun()

if __name__ == "__main__":
    main()
