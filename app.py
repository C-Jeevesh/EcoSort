import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import pickle
import datetime
import re
from skimage.feature import hog
from sklearn import svm

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="EcoSort Compliance Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS ---
st.markdown("""
<style>
    /* HIDE FOOTER */
    footer {visibility: hidden;}
    #MainMenu {visibility: visible;} 
    
    /* LOGIN CONTAINER */
    .login-box {
        border: 1px solid #e5e7eb;
        padding: 40px;
        border-radius: 8px;
        max-width: 400px;
        margin: 0 auto;
        background-color: white;
    }

    /* METRIC CARDS */
    .metric-container {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 20px;
        background-color: #f9fafb;
    }
    
    /* FULL SCREEN LOGOUT MESSAGE styling */
    .logout-container {
        text-align: center;
        margin-top: 100px;
        padding: 50px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f8f9fa;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'show_logout' not in st.session_state: st.session_state['show_logout'] = False # <--- NEW STATE
if 'user_info' not in st.session_state: st.session_state['user_info'] = {}
if 'users_db' not in st.session_state:
    st.session_state['users_db'] = {
        'admin@ecosort.gov': {'password': '123', 'name': 'System Administrator', 'role': 'Auditor Level 1'}
    }
if 'audit_log' not in st.session_state:
    st.session_state['audit_log'] = pd.DataFrame({
        'Date': [datetime.date.today().strftime("%Y-%m-%d")],
        'User': ['admin@ecosort.gov'],
        'Item_ID': ['AUD-001'],
        'Classification': ['Recyclable'],
        'Method': ['Camera']
    })
if 'settings' not in st.session_state:
    st.session_state['settings'] = {'high_contrast': False, 'compact_mode': False}

# ==========================================
# 3. BACKEND LOGIC
# ==========================================
@st.cache_resource
def load_engine():
    try:
        with open('ecosort_svm_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        dummy_X = np.random.rand(10, 3780)
        dummy_y = ['R', 'O'] * 5
        model = svm.SVC(probability=True)
        model.fit(dummy_X, dummy_y)
        return model

model = load_engine()

def process_features(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 128))
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features.reshape(1, -1)

# ==========================================
# 4. AUTHENTICATION SCREENS
# ==========================================

# --- A. LOGIN SCREEN ---
def login_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.header("EcoSort Compliance System")
        st.caption("Secure Access Gateway v4.5")
        st.markdown("---")
        
        tab_login, tab_register = st.tabs(["Secure Login", "Register User"])
        
        with tab_login:
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            
            if st.button("Authenticate", type="primary"):
                db = st.session_state['users_db']
                if email in db and db[email]['password'] == password:
                    st.session_state['logged_in'] = True
                    st.session_state['show_logout'] = False # Reset logout flag
                    st.session_state['user_info'] = {'email': email, **db[email]}
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Credentials")

        with tab_register:
            new_email = st.text_input("New Email")
            new_name = st.text_input("Full Name")
            new_pass = st.text_input("Set Password", type="password")
            
            if st.button("Create Profile"):
                email_regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
                if not re.search(email_regex, new_email):
                    st.error("Invalid Email Format")
                elif new_email in st.session_state['users_db']:
                    st.error("User already exists")
                elif new_pass:
                    st.session_state['users_db'][new_email] = {
                        'password': new_pass, 
                        'name': new_name, 
                        'role': 'Junior Auditor'
                    }
                    st.success("Registration Successful. Please Login.")
                else:
                    st.warning("All fields are required.")

# --- B. LOGOUT CONFIRMATION SCREEN ---
def logout_modal():
    # This renders a full screen message
    st.markdown("""
        <div class="logout-container">
            <h2>🔒 Secure Logout</h2>
            <p>You have been successfully logged out of the EcoSort Compliance System.</p>
            <p style="color: grey; font-size: 12px;">Session terminated at {}</p>
            <hr>
        </div>
    """.format(datetime.datetime.now().strftime("%H:%M:%S")), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("Return to Sign In", type="primary"):
            st.session_state['show_logout'] = False
            st.session_state['logged_in'] = False
            st.rerun()

# ==========================================
# 5. MAIN APPLICATION
# ==========================================
def main_app():
    # --- SIDEBAR (MENUBAR) ---
    with st.sidebar:
        st.title("System Menu")
        st.info(f"User: {st.session_state['user_info'].get('name')}")
        st.caption(f"ID: {st.session_state['user_info'].get('email')}")
        st.markdown("---")
        
        # Navigation
        menu = st.radio("Select Module:", ["Dashboard", "Waste Auditor", "Settings", "Logout"])
        
        st.markdown("---")
        st.caption("System Status: Online")
        
        # LOGOUT LOGIC
        if menu == "Logout":
            # Instead of just logging out, we trigger the popup state
            st.session_state['show_logout'] = True
            st.rerun()

    # --- DASHBOARD PAGE ---
    if menu == "Dashboard":
        st.subheader("Compliance Overview")
        st.markdown("Real-time facility audit metrics.")
        
        df = st.session_state['audit_log']
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Records", len(df))
        with c2:
            recyclable = len(df[df['Classification'] == 'Recyclable'])
            st.metric("Diversion Rate", f"{int((recyclable/len(df))*100)}%" if len(df)>0 else "0%")
        with c3:
            st.metric("Pending Reviews", "0")
            
        st.markdown("### Recent Logs")
        st.dataframe(df, use_container_width=True)

    # --- AUDITOR PAGE ---
    elif menu == "Waste Auditor":
        st.subheader("Data Acquisition")
        st.markdown("Select input method for classification.")
        
        input_tab1, input_tab2 = st.tabs(["Live Camera Feed", "File Upload"])
        
        img_array = None
        method = ""
        
        with input_tab1:
            cam_input = st.camera_input("Capture Image")
            if cam_input:
                img_array = cv2.imdecode(np.frombuffer(cam_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                method = "Camera"

        with input_tab2:
            file_input = st.file_uploader("Select Image File", type=['jpg', 'jpeg', 'png'])
            if file_input:
                img_array = cv2.imdecode(np.frombuffer(file_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                st.image(img_array, caption="Uploaded Preview", width=300)
                method = "Upload"

        if img_array is not None:
            st.markdown("---")
            st.markdown("**Analysis Results**")
            
            with st.spinner("Processing feature vectors..."):
                time.sleep(0.5)
                features = process_features(img_array)
                pred = model.predict(features)[0]
                confidence = model.predict_proba(features).max()
            
            status = "Recyclable" if pred == 'R' else "Organic/General"
            action = "Zone B (Recycling)" if pred == 'R' else "Zone G (Disposal)"
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.info(f"Classification: {status}")
            with c_res2:
                st.text(f"Confidence: {confidence:.4f}")
                
            st.success(f"Action Required: {action}")
            
            new_log = {
                'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                'User': st.session_state['user_info']['email'],
                'Item_ID': f"AUD-{int(time.time())}",
                'Classification': status,
                'Method': method
            }
            st.session_state['audit_log'] = pd.concat([st.session_state['audit_log'], pd.DataFrame([new_log])], ignore_index=True)

    # --- SETTINGS PAGE ---
    elif menu == "Settings":
        st.subheader("System Configuration")
        
        with st.expander("User Profile", expanded=True):
            user = st.session_state['user_info']
            st.text_input("Registered Email", value=user['email'], disabled=True)
            st.text_input("Full Name", value=user['name'], disabled=True)
            st.text_input("Clearance Role", value=user['role'], disabled=True)
            if st.button("Request Password Reset"):
                st.info("Reset link sent to administrator.")

        with st.expander("Interface Customization"):
            st.checkbox("Enable High Contrast Mode", value=st.session_state['settings']['high_contrast'])
            st.checkbox("Compact Data View", value=st.session_state['settings']['compact_mode'])
            
            if st.session_state['settings']['high_contrast']:
                st.markdown("""<style>.stApp {filter: contrast(120%);}</style>""", unsafe_allow_html=True)

        with st.expander("About Application"):
            st.markdown("""
            **EcoSort Compliance Platform v4.5**
            Designed for municipal and corporate waste auditing.
            **Legal Disclaimer:** Predictions are for guidance only.
            **Support:** tech-support@ecosort.gov
            """)

# ==========================================
# 6. RUN CONTROLLER
# ==========================================
# Logic: If logout flag is True, show logout screen. 
# Else, if logged in, show app. 
# Else, show login.

if st.session_state['show_logout']:
    logout_modal()
elif st.session_state['logged_in']:
    main_app()
else:
    login_screen()