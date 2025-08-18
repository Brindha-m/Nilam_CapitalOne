import streamlit as st
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Nilam - Your Agricultural Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with improved selectbox styling
st.markdown("""
<style>
    :root {
        --cream-light: #fefcf8;
        --cream-medium: #f5f1e8;
        --cream-dark: #ede4d3;
        --earth-green: #7a8471;
        --earth-brown: #8b7355;
        --text-primary: #2d3436;
        --text-secondary: #636e72;
        --graph-text: #2d3436;
        --graph-text-secondary: #636e72;
        --sidebar-text: #ffffff;
        --selection-text: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--cream-light) 0%, var(--cream-medium) 100%);
    }


    .main-header {
        background: linear-gradient(135deg, var(--earth-brown) 0%, var(--cream-dark) 50%, var(--earth-green) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: black;  /* Changed to black */
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.2);
        font-size: 8rem;
        text-shadow: none;
    }

    
    .expert-response-container {
        background: linear-gradient(135deg, var(--cream-light) 0%, var(--cream-medium) 100%);
        border: 3px solid var(--earth-green);
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(122, 132, 113, 0.2);
        overflow: hidden;
    }
    
    .response-header {
        background: linear-gradient(135deg, var(--earth-green) 0%, var(--earth-brown) 100%);
        color: white;
        padding: 1.5rem 2rem;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        border-bottom: 3px solid var(--earth-brown);
    }
    
    .response-content {
        padding: 2.5rem;
        color: var(--text-primary);
        line-height: 1.8;
        font-size: 16px;
        text-align: justify;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .response-content .section-header {
        background: linear-gradient(135deg, var(--earth-brown) 0%, var(--earth-green) 100%);
        color: white !important;
        padding: 1.2rem 2rem;
        margin: 2.5rem -2.5rem 2rem -2.5rem;
        font-size: 1.4rem !important;
        font-weight: bold !important;
        text-align: center;
        border-radius: 0;
        box-shadow: 0 4px 15px rgba(139, 115, 85, 0.3);
    }
    
    .bullet-point {
        background: linear-gradient(135deg, var(--cream-medium) 0%, var(--cream-dark) 100%);
        margin: 0.8rem 0;
        padding: 1rem 1.5rem;
        border-left: 5px solid var(--earth-green);
        border-radius: 0 10px 10px 0;
        box-shadow: 0 3px 10px rgba(122, 132, 113, 0.15);
        font-weight: 500;
        transition: all 0.3s ease;
        text-align: left;
        line-height: 1.6;
    }
    
    .bullet-point:hover {
        transform: translateX(10px);
        box-shadow: 0 5px 15px rgba(122, 132, 113, 0.25);
    }
    
    .important-note {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #f39c12;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 2rem 0;
        box-shadow: 0 6px 20px rgba(243, 156, 18, 0.2);
        font-weight: 600;
        color: #856404;
        text-align: left;
        line-height: 1.6;
    }
    
    .response-table {
        margin: 2rem 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        background: white;
    }
    
    .response-table table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 0 !important;
    }
    
    .response-table th {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%) !important;
        color: white !important;
        font-weight: bold !important;
        padding: 1.2rem !important;
        text-align: center !important;
        font-size: 1rem !important;
        border-bottom: 3px solid var(--earth-green) !important;
    }
    
    .response-table td {
        padding: 1rem !important;
        border: 1px solid #e0e0e0 !important;
        color: #2d3436 !important;
        font-size: 0.95rem !important;
        vertical-align: middle !important;
        text-align: center !important;
    }
    
    .response-table tr:nth-child(even) {
        background: linear-gradient(135deg, #f9f9f9 0%, #f5f5f5 100%) !important;
    }
    
    .response-table tr:hover {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%) !important;
        transform: scale(1.01);
        transition: all 0.3s ease !important;
    }
    
    .response-section-card {
        background: linear-gradient(135deg, var(--cream-light) 0%, var(--cream-medium) 100%);
        border: 2px solid var(--earth-green);
        border-radius: 15px;
        margin: 2rem 0;
        padding: 0;
        box-shadow: 0 8px 25px rgba(122, 132, 113, 0.2);
        overflow: hidden;
    }
    
    .card-header {
        background: linear-gradient(135deg, var(--earth-green) 0%, var(--earth-brown) 100%);
        color: white;
        padding: 1.2rem 2rem;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        border-bottom: 3px solid var(--earth-brown);
    }
    
    .card-content {
        padding: 2rem;
        color: var(--text-primary);
        line-height: 1.7;
        text-align: justify;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--cream-medium) 0%, var(--cream-dark) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--earth-brown);
        text-align: center;
        color: var(--text-primary);
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(139, 115, 85, 0.15);
        transition: transform 0.3s ease;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, var(--cream-light) 0%, var(--cream-medium) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid var(--earth-green);
        margin: 1rem 0;
        color: var(--text-primary);
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(122, 132, 113, 0.15);
        text-align: left;
        line-height: 1.6;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--earth-green) 0%, var(--earth-brown) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(122, 132, 113, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(122, 132, 113, 0.4);
    }
    
    .stButton > button[data-testid*="enhanced_quick_"] {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%) !important;
        color: white !important;
        border: 2px solid #7a8471 !important;
        font-size: 13px !important;
        padding: 0.8rem 1rem !important;
        margin: 0.4rem 0 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button[data-testid*="enhanced_quick_"]:hover {
        background: linear-gradient(135deg, #7a8471 0%, #8b7355 100%) !important;
        transform: translateX(5px) !important;
        box-shadow: 0 4px 12px rgba(122, 132, 113, 0.3) !important;
    }
    
    h1, h2, h3, h4 { 
        color: var(--text-primary) !important; 
        font-weight: 700;
    }
    
    /* Increase h1 font size */
    h1 {
        font-size: 2.5rem !important;
        line-height: 1.2 !important;
    }
    
    /* Large h1 header styling */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        background: linear-gradient(135deg, var(--earth-brown) 0%, var(--cream-dark) 50%, var(--earth-green) 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        padding: 1rem 0 !important;
    }
    
    p, li, span, div { 
        color: var(--text-primary) !important; 
        font-size: 16px;
    }
    
    /* Main app specific styles */
    .section-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
            
            /* Sidebar Selectbox label */
div[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 600 !important;
}

/* Selected value inside the selectbox */
div[data-testid="stSidebar"] .stSelectbox div[role="combobox"] input {
    color: white !important;
    font-weight: 600 !important;
}

/* Dropdown arrow icon */
div[data-testid="stSidebar"] .stSelectbox svg {
    fill: white !important;
}

/* Dropdown option text */
div[data-testid="stSidebar"] .stSelectbox div[role="listbox"] div {
    color: white !important;
}

/* Comprehensive dropdown list styling */
[data-testid="stSidebar"] .stSelectbox div[role="listbox"] *,
[data-testid="stSidebar"] .stSelectbox div[role="listbox"] div,
[data-testid="stSidebar"] .stSelectbox div[role="listbox"] span,
[data-testid="stSidebar"] .stSelectbox div[role="listbox"] p,
[data-testid="stSidebar"] .stSelectbox div[role="listbox"] li,
[data-testid="stSidebar"] .stSelectbox div[role="listbox"] ul,
[data-testid="stSidebar"] .stSelectbox div[role="listbox"] ol {
    color: white !important;
    background-color: #2d3436 !important;
}

/* Dropdown popup styling */
[data-testid="stSidebar"] div[data-baseweb="popover"] *,
[data-testid="stSidebar"] div[data-baseweb="popover"] div,
[data-testid="stSidebar"] div[data-baseweb="popover"] span,
[data-testid="stSidebar"] div[data-baseweb="popover"] p,
[data-testid="stSidebar"] div[data-baseweb="popover"] li,
[data-testid="stSidebar"] div[data-baseweb="popover"] ul,
[data-testid="stSidebar"] div[data-baseweb="popover"] ol {
    color: white !important;
    background-color: #2d3436 !important;
}

/* Force white text for all dropdown elements */
[data-testid="stSidebar"] .stSelectbox *,
[data-testid="stSidebar"] .stSelectbox div *,
[data-testid="stSidebar"] .stSelectbox div div *,
[data-testid="stSidebar"] .stSelectbox div div div * {
    color: white !important;
}

/* Ultra comprehensive dropdown styling - force white text */
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stSelectbox *,
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stSelectbox > div > div > div,
[data-testid="stSidebar"] .stSelectbox span,
[data-testid="stSidebar"] .stSelectbox div span,
[data-testid="stSidebar"] .stSelectbox div div span,
[data-testid="stSidebar"] .stSelectbox div div div span,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSelectbox div label,
[data-testid="stSidebar"] .stSelectbox div div label,
[data-testid="stSidebar"] .stSelectbox div div div label,
[data-testid="stSidebar"] .stSelectbox p,
[data-testid="stSidebar"] .stSelectbox div p,
[data-testid="stSidebar"] .stSelectbox div div p,
[data-testid="stSidebar"] .stSelectbox div div div p,
[data-testid="stSidebar"] .stSelectbox strong,
[data-testid="stSidebar"] .stSelectbox div strong,
[data-testid="stSidebar"] .stSelectbox div div strong,
[data-testid="stSidebar"] .stSelectbox div div div strong,
[data-testid="stSidebar"] .stSelectbox em,
[data-testid="stSidebar"] .stSelectbox div em,
[data-testid="stSidebar"] .stSelectbox div div em,
[data-testid="stSidebar"] .stSelectbox div div div em,
[data-testid="stSidebar"] .stSelectbox b,
[data-testid="stSidebar"] .stSelectbox div b,
[data-testid="stSidebar"] .stSelectbox div div b,
[data-testid="stSidebar"] .stSelectbox div div div b,
[data-testid="stSidebar"] .stSelectbox i,
[data-testid="stSidebar"] .stSelectbox div i,
[data-testid="stSidebar"] .stSelectbox div div i,
[data-testid="stSidebar"] .stSelectbox div div div i {
    color: white !important;
}

/* Force white text for all possible dropdown selectors */
[data-testid="stSidebar"] div[data-baseweb="select"] *,
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] div,
[data-testid="stSidebar"] div[data-baseweb="select"] div span,
[data-testid="stSidebar"] div[data-baseweb="select"] div div,
[data-testid="stSidebar"] div[data-baseweb="select"] div div span,
[data-testid="stSidebar"] div[data-baseweb="select"] div div div,
[data-testid="stSidebar"] div[data-baseweb="select"] div div div span,
[data-testid="stSidebar"] div[data-baseweb="select"] label,
[data-testid="stSidebar"] div[data-baseweb="select"] div label,
[data-testid="stSidebar"] div[data-baseweb="select"] div div label,
[data-testid="stSidebar"] div[data-baseweb="select"] div div div label,
[data-testid="stSidebar"] div[data-baseweb="select"] p,
[data-testid="stSidebar"] div[data-baseweb="select"] div p,
[data-testid="stSidebar"] div[data-baseweb="select"] div div p,
[data-testid="stSidebar"] div[data-baseweb="select"] div div div p {
    color: white !important;
}

/* Force white text for dropdown options specifically */
[data-testid="stSidebar"] .stSelectbox option,
[data-testid="stSidebar"] .stSelectbox select option,
[data-testid="stSidebar"] div[data-baseweb="select"] option,
[data-testid="stSidebar"] div[data-baseweb="select"] select option,
[data-testid="stSidebar"] [role="option"],
[data-testid="stSidebar"] [role="option"] *,
[data-testid="stSidebar"] .stSelectbox [role="option"],
[data-testid="stSidebar"] .stSelectbox [role="option"] *,
[data-testid="stSidebar"] div[data-baseweb="select"] [role="option"],
[data-testid="stSidebar"] div[data-baseweb="select"] [role="option"] * {
    color: white !important;
    background-color: #2d3436 !important;
}

/* Force white text for all text elements in sidebar */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Make selectbox labels black - Ultra comprehensive */
.stSelectbox label,
div[data-baseweb="select"] label,
[data-testid="stSelectbox"] label {
    color: black !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* Force black color for all possible label selectors */
label,
.stSelectbox label,
div[data-baseweb="select"] label,
[data-testid="stSelectbox"] label,
.stSelectbox div label,
div[data-baseweb="select"] div label,
[data-testid="stSelectbox"] div label,
.stSelectbox div div label,
div[data-baseweb="select"] div div label,
[data-testid="stSelectbox"] div div label,
.stSelectbox div div div label,
div[data-baseweb="select"] div div div label,
[data-testid="stSelectbox"] div div div label {
    color: black !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    text-shadow: none !important;
    opacity: 1 !important;
}

/* Ultra specific targeting for selectbox labels */
.stSelectbox > div > label,
div[data-baseweb="select"] > div > label,
[data-testid="stSelectbox"] > div > label,
.stSelectbox > div > div > label,
div[data-baseweb="select"] > div > div > label,
[data-testid="stSelectbox"] > div > div > label {
    color: black !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    text-shadow: none !important;
    opacity: 1 !important;
}

/* Force black color for all label elements */
* label {
    color: black !important;
    font-weight: 600 !important;
    text-shadow: none !important;
    opacity: 1 !important;
}

/* Force white text for all selectbox elements in main content */
.stSelectbox *,
.stSelectbox label,
.stSelectbox span,
.stSelectbox div,
.stSelectbox div span,
.stSelectbox div div,
.stSelectbox div div span,
.stSelectbox div div div,
.stSelectbox div div div span,
.stSelectbox option,
.stSelectbox select,
.stSelectbox select option {
    color: white !important;
}

/* Force white text for all baseweb select elements */
div[data-baseweb="select"] *,
div[data-baseweb="select"] label,
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] div span,
div[data-baseweb="select"] div div,
div[data-baseweb="select"] div div span,
div[data-baseweb="select"] div div div,
div[data-baseweb="select"] div div div span,
div[data-baseweb="select"] option,
div[data-baseweb="select"] select,
div[data-baseweb="select"] select option {
    color: white !important;
}

/* Force white text for all testid selectbox elements */
[data-testid="stSelectbox"] *,
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div,
[data-testid="stSelectbox"] div span,
[data-testid="stSelectbox"] div div,
[data-testid="stSelectbox"] div div span,
[data-testid="stSelectbox"] div div div,
[data-testid="stSelectbox"] div div div span,
[data-testid="stSelectbox"] option,
[data-testid="stSelectbox"] select,
[data-testid="stSelectbox"] select option {
    color: white !important;
}

/* Force white text for dropdown popup content */
div[data-baseweb="popover"] *,
div[data-baseweb="popover"] label,
div[data-baseweb="popover"] span,
div[data-baseweb="popover"] div,
div[data-baseweb="popover"] div span,
div[data-baseweb="popover"] div div,
div[data-baseweb="popover"] div div span,
div[data-baseweb="popover"] div div div,
div[data-baseweb="popover"] div div div span,
div[data-baseweb="popover"] option,
div[data-baseweb="popover"] select,
div[data-baseweb="popover"] select option {
    color: white !important;
    background-color: #2d3436 !important;
}

/* Force white text for role option elements */
[role="option"] *,
[role="option"] label,
[role="option"] span,
[role="option"] div,
[role="option"] div span,
[role="option"] div div,
[role="option"] div div span,
[role="option"] div div div,
[role="option"] div div div span {
    color: white !important;
    background-color: #2d3436 !important;
}

/* Ultra comprehensive selectbox styling for main content */
.stSelectbox,
.stSelectbox *,
.stSelectbox > div,
.stSelectbox > div > div,
.stSelectbox > div > div > div,
.stSelectbox span,
.stSelectbox div span,
.stSelectbox div div span,
.stSelectbox div div div span,
.stSelectbox p,
.stSelectbox div p,
.stSelectbox div div p,
.stSelectbox div div div p,
.stSelectbox strong,
.stSelectbox div strong,
.stSelectbox div div strong,
.stSelectbox div div div strong,
.stSelectbox em,
.stSelectbox div em,
.stSelectbox div div em,
.stSelectbox div div div em,
.stSelectbox b,
.stSelectbox div b,
.stSelectbox div div b,
.stSelectbox div div div b,
.stSelectbox i,
.stSelectbox div i,
.stSelectbox div div i,
.stSelectbox div div div i {
    color: white !important;
}



/* Force white text for all sidebar titles and text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] em,
[data-testid="stSidebar"] b,
[data-testid="stSidebar"] i,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] ul,
[data-testid="stSidebar"] ol {
    color: white !important;
}

/* Ultra comprehensive sidebar text styling */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] * *,
[data-testid="stSidebar"] * * * {
    color: white !important;
}

/* Specific sidebar title styling */
[data-testid="stSidebar"] .css-1d391kg,
[data-testid="stSidebar"] .css-1lcbmhc,
[data-testid="stSidebar"] .css-1v0mbdj,
[data-testid="stSidebar"] [data-testid="stSidebar"] {
    color: white !important;
}

            
</style>
""", unsafe_allow_html=True)

def run_nilamchat():
    """Run nilamchat functionality directly"""
    try:
        # Add the current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        # Import nilamchat module
        import nilamchat
        
        # Run the nilamchat main function
        if hasattr(nilamchat, 'main'):
            nilamchat.main()
        else:
            st.error("No main function found in nilamchat.py")
            
    except Exception as e:
        st.error(f"Error loading Nilam Chat: {str(e)}")
        # st.info("Please ensure nilamchat.py is in the same directory.")

def run_leafine():
    """Run leafine functionality"""
    try:
        # Add the current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        # Import leafine module
        import leafine
        
        # Run the leafine main function
        if hasattr(leafine, 'main'):
            leafine.main()
        else:
            st.error("No main function found in leafine.py")
            
    except Exception as e:
        st.error(f"Error loading Leafine: {str(e)}")
        st.info("Please ensure leafine.py is in the same directory.")

def run_nilamsense():
    """Run nilamsense functionality"""
    try:
        # Add the current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        # Import nilamsense module
        import nilamsense
        
        # Create and run the CropRecommendationApp
        app = nilamsense.CropRecommendationApp()
        app.run()
            
    except Exception as e:
        st.error(f"Error loading NilamSense: {str(e)}")
        st.info("Please ensure nilamsense.py is in the same directory.")

def main():
    # Sidebar navigation
    st.sidebar.title("🌱 Nilam Navigation")
    st.sidebar.markdown("---")
    
    # Navigation options
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Nilam Chat", "Leafine", "NilamSense"],
        index=0  # Default to Nilam Chat
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **Nilam** - Your Agricultural Assistant
    
    Navigate between different sections:
    - **Nilam Chat**: AI-powered agricultural assistance
    - **Leafine**: Leaf disease detection
    - **NilamSense**: Agricultural intelligence & insights
    """)
    
    # Main content area
    if page == "Nilam Chat": 
        run_nilamchat()
    
    elif page == "Leafine":
        st.markdown("""
            <div class='main-header'>
                    <div style='font-size: 1.8rem; opacity: 0.9;'>
                        🍃 Leafine
                    </div>
            </div>
        """, unsafe_allow_html=True)
        run_leafine()
    
    elif page == "NilamSense":
        st.markdown("""
            <div class='main-header'>
                    <div style='font-size: 1.8rem; opacity: 0.9;'>
                        🧠 NilamSense
                    </div>
            </div>
        """, unsafe_allow_html=True)
        run_nilamsense()

if __name__ == "__main__":
    main()
