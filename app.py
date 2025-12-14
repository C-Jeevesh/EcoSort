import streamlit as st
import cv2
import numpy as np
import pickle
import time
import pandas as pd
from skimage.feature import hog

# --- 1. APP CONFIGURATION (Responsive Layout) ---
st.set_page_config(
    page_title="EcoSort Global",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MULTI-LANGUAGE DICTIONARY (7 Languages) ---
TRANSLATIONS = {
    "English": {
        "nav_dash": "Dashboard", "nav_scan": "Scanner", "nav_hist": "History", "nav_set": "Settings",
        "welcome": "Welcome, Eco-Warrior!", "stats_title": "Your Daily Impact",
        "metric_scan": "Total Scans", "metric_score": "Eco Points", "metric_lvl": "Current Rank",
        "rank_name": "Recycling Rookie", "rank_gold": "Green Guardian 🛡️",
        "scan_head": "AI Waste Identifier", "cam_btn": "Use Camera", "up_btn": "Upload File",
        "analyze_btn": "Analyze Waste", "analyzing": "Processing image...",
        "res_recycle": "♻️ RECYCLABLE", "res_organic": "🍎 ORGANIC / TRASH",
        "act_recycle": "Clean it and place in the Blue Bin.",
        "act_organic": "Compost or place in General Waste.",
        "toast_win": "+10 Points! Added to history.",
        "hist_head": "Recent Scans", "hist_empty": "No items scanned yet.",
        "set_head": "App Settings", "set_dark": "Dark Mode", "set_notif": "Notifications"
    },
    "Hindi (हिंदी)": {
        "nav_dash": "डैशबोर्ड", "nav_scan": "स्कैनर", "nav_hist": "इतिहास", "nav_set": "सेटिंग्स",
        "welcome": "स्वागत है, पर्यावरण रक्षक!", "stats_title": "आपका आज का प्रभाव",
        "metric_scan": "कुल स्कैन", "metric_score": "इको पॉइंट्स", "metric_lvl": "वर्तमान रैंक",
        "rank_name": "नया रक्षक", "rank_gold": "ग्रीन गार्डियन 🛡️",
        "scan_head": "AI कचरा पहचानकर्ता", "cam_btn": "कैमरा", "up_btn": "फाइल अपलोड",
        "analyze_btn": "विश्लेषण करें", "analyzing": "प्रोसेसिंग हो रही है...",
        "res_recycle": "♻️ रिसाइकिल योग्य", "res_organic": "🍎 जैविक / कचरा",
        "act_recycle": "साफ करें और नीले डिब्बे में डालें।",
        "act_organic": "खाद बनाएं या सामान्य कचरे में डालें।",
        "toast_win": "+10 अंक! इतिहास में जोड़ा गया।",
        "hist_head": "हाल ही के स्कैन", "hist_empty": "अभी तक कुछ भी स्कैन नहीं किया गया।",
        "set_head": "ऐप सेटिंग्स", "set_dark": "डार्क मोड", "set_notif": "सूचनाएं"
    },
    "Spanish (Español)": {
        "nav_dash": "Tablero", "nav_scan": "Escáner", "nav_hist": "Historial", "nav_set": "Ajustes",
        "welcome": "¡Bienvenido, Guerrero Eco!", "stats_title": "Tu Impacto Diario",
        "metric_scan": "Escaneos", "metric_score": "Puntos Eco", "metric_lvl": "Rango Actual",
        "rank_name": "Principiante", "rank_gold": "Guardián Verde 🛡️",
        "scan_head": "Identificador IA", "cam_btn": "Usar Cámara", "up_btn": "Subir Archivo",
        "analyze_btn": "Analizar", "analyzing": "Procesando...",
        "res_recycle": "♻️ RECICLABLE", "res_organic": "🍎 ORGÁNICO / BASURA",
        "act_recycle": "Limpiar y colocar en el contenedor azul.",
        "act_organic": "Compost o basura general.",
        "toast_win": "¡+10 Puntos! Agregado al historial.",
        "hist_head": "Escaneos Recientes", "hist_empty": "Nada escaneado aún.",
        "set_head": "Configuración", "set_dark": "Modo Oscuro", "set_notif": "Notificaciones"
    },
    "French (Français)": {
        "nav_dash": "Tableau de bord", "nav_scan": "Scanner", "nav_hist": "Historique", "nav_set": "Paramètres",
        "welcome": "Bienvenue, Éco-Guerrier!", "stats_title": "Votre Impact",
        "metric_scan": "Scans Totaux", "metric_score": "Eco Points", "metric_lvl": "Rang Actuel",
        "rank_name": "Débutant", "rank_gold": "Gardien Vert 🛡️",
        "scan_head": "Identificateur IA", "cam_btn": "Caméra", "up_btn": "Télécharger",
        "analyze_btn": "Analyser", "analyzing": "Traitement...",
        "res_recycle": "♻️ RECYCLABLE", "res_organic": "🍎 ORGANIQUE / DÉCHET",
        "act_recycle": "Nettoyer et placer dans le bac bleu.",
        "act_organic": "Compost ou poubelle générale.",
        "toast_win": "+10 Points! Ajouté à l'historique.",
        "hist_head": "Scans Récents", "hist_empty": "Aucun scan.",
        "set_head": "Paramètres", "set_dark": "Mode Sombre", "set_notif": "Notifications"
    },
    "German (Deutsch)": {
        "nav_dash": "Instrumententafel", "nav_scan": "Scanner", "nav_hist": "Verlauf", "nav_set": "Einstellungen",
        "welcome": "Willkommen, Öko-Krieger!", "stats_title": "Dein Einfluss",
        "metric_scan": "Gesamtscans", "metric_score": "Öko-Punkte", "metric_lvl": "Aktueller Rang",
        "rank_name": "Anfänger", "rank_gold": "Grüner Wächter 🛡️",
        "scan_head": "KI-Abfallscanner", "cam_btn": "Kamera", "up_btn": "Hochladen",
        "analyze_btn": "Analysieren", "analyzing": "Verarbeitung...",
        "res_recycle": "♻️ RECYCELBAR", "res_organic": "🍎 BIO / MÜLL",
        "act_recycle": "Reinigen und in die blaue Tonne geben.",
        "act_organic": "Kompost oder Restmüll.",
        "toast_win": "+10 Punkte! Zum Verlauf hinzugefügt.",
        "hist_head": "Letzte Scans", "hist_empty": "Noch keine Scans.",
        "set_head": "Einstellungen", "set_dark": "Dunkelmodus", "set_notif": "Benachrichtigungen"
    },
    "Mandarin (中文)": {
        "nav_dash": "仪表板 (Dashboard)", "nav_scan": "扫描仪 (Scanner)", "nav_hist": "历史 (History)", "nav_set": "设置 (Settings)",
        "welcome": "欢迎, 环保卫士!", "stats_title": "你的日常影响",
        "metric_scan": "总扫描数", "metric_score": "环保积分", "metric_lvl": "当前等级",
        "rank_name": "新手", "rank_gold": "绿色守护者 🛡️",
        "scan_head": "AI 垃圾识别", "cam_btn": "使用相机", "up_btn": "上传图片",
        "analyze_btn": "开始分析", "analyzing": "处理中...",
        "res_recycle": "♻️ 可回收", "res_organic": "🍎 有机 / 垃圾",
        "act_recycle": "清洗并放入蓝色垃圾桶。",
        "act_organic": "堆肥或放入普通垃圾桶。",
        "toast_win": "+10 分! 已添加到历史记录。",
        "hist_head": "最近扫描", "hist_empty": "暂无记录。",
        "set_head": "设置", "set_dark": "深色模式", "set_notif": "通知"
    },
    "Japanese (日本語)": {
        "nav_dash": "ダッシュボード", "nav_scan": "スキャナー", "nav_hist": "履歴", "nav_set": "設定",
        "welcome": "ようこそ、エコ戦士！", "stats_title": "毎日の影響",
        "metric_scan": "スキャン総数", "metric_score": "エコポイント", "metric_lvl": "現在のランク",
        "rank_name": "ルーキー", "rank_gold": "グリーンガーディアン 🛡️",
        "scan_head": "AI ゴミ識別", "cam_btn": "カメラ", "up_btn": "アップロード",
        "analyze_btn": "分析する", "analyzing": "処理中...",
        "res_recycle": "♻️ リサイクル可能", "res_organic": "🍎 生ゴミ / その他",
        "act_recycle": "洗って青い箱に入れてください。",
        "act_organic": "堆肥または一般ゴミ。",
        "toast_win": "+10 ポイント！履歴に追加されました。",
        "hist_head": "最近のスキャン", "hist_empty": "まだスキャンはありません。",
        "set_head": "設定", "set_dark": "ダークモード", "set_notif": "通知"
    }
}

# --- 3. SESSION STATE & HELPERS ---
if 'lang' not in st.session_state: st.session_state['lang'] = 'English'
if 'score' not in st.session_state: st.session_state['score'] = 0
if 'history' not in st.session_state: st.session_state['history'] = []

def txt(key):
    return TRANSLATIONS[st.session_state['lang']][key]

# --- 4. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        with open('ecosort_svm_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
model = load_model()

# --- 5. IMAGE PROCESSING ---
def process_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 128))
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features.reshape(1, -1)

# --- 6. SIDEBAR UI ---
with st.sidebar:
    st.title("🌍 EcoSort Global")
    
    # Language Dropdown
    selected_lang = st.selectbox("🌐 Language / भाषा / 言語", list(TRANSLATIONS.keys()))
    st.session_state['lang'] = selected_lang
    
    st.divider()
    
    # Navigation Buttons
    page = st.radio("Menu", [txt('nav_dash'), txt('nav_scan'), txt('nav_hist'), txt('nav_set')], label_visibility="collapsed")
    
    st.divider()
    
    # Mini Stats in Sidebar
    st.metric("🏆 " + txt('metric_score'), st.session_state['score'])

# --- 7. PAGE ROUTING ---

# === DASHBOARD ===
if page == txt('nav_dash'):
    st.title(txt('welcome'))
    st.markdown("---")
    
    # 3-Column Layout for Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**{txt('metric_scan')}**")
        st.subheader(f"{len(st.session_state['history'])}")

    with col2:
        st.success(f"**{txt('metric_score')}**")
        st.subheader(f"{st.session_state['score']}")

    with col3:
        rank = txt('rank_name')
        if st.session_state['score'] > 50: rank = txt('rank_gold')
        st.warning(f"**{txt('metric_lvl')}**")
        st.subheader(rank)
        
    st.markdown("### 📈 Activity Trend")
    # Fake chart for visual appeal
    chart_data = pd.DataFrame(np.random.randn(7, 1), columns=["Impact"])
    st.line_chart(chart_data)

# === SCANNER ===
elif page == txt('nav_scan'):
    st.header(txt('scan_head'))

    if model is None:
        st.error("⚠️ Model file missing. Please check your folder.")
        st.stop()
    
    # Input Tabs (Smoother than radio buttons)
    tab1, tab2 = st.tabs([f"📸 {txt('cam_btn')}", f"📂 {txt('up_btn')}"])
    
    img_input = None
    
    with tab1:
        cam_img = st.camera_input("Camera")
        if cam_img: img_input = cam_img
            
    with tab2:
        up_img = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])
        if up_img: img_input = up_img

    if img_input:
        # Convert image
        file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Display nicely centered
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(opencv_image, channels="BGR", use_container_width=True, caption="Input Preview")
            
            if st.button(txt('analyze_btn'), type="primary", use_container_width=True):
                with st.spinner(txt('analyzing')):
                    time.sleep(0.8) # UX Feel
                    
                    features = process_image(opencv_image)
                    pred = model.predict(features)[0]
                    prob = model.predict_proba(features).max()
                    
                    # Logic
                    result_text = txt('res_recycle') if pred == 'R' else txt('res_organic')
                    action_text = txt('act_recycle') if pred == 'R' else txt('act_organic')
                    color = "green" if pred == 'R' else "orange"
                    
                    # Update State
                    st.session_state['score'] += 10
                    st.session_state['history'].insert(0, {"result": result_text, "conf": f"{prob*100:.1f}%", "time": time.strftime("%H:%M")})
                    
                    st.toast(txt('toast_win'), icon="🎉")
                
                # Result Card
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {color};">
                    <h2 style="color: black; margin:0;">{result_text}</h2>
                    <p style="color: gray; margin:0;">Confidence: {prob*100:.1f}%</p>
                    <hr>
                    <p style="color: black; font-size: 18px;">💡 {action_text}</p>
                </div>
                """, unsafe_allow_html=True)

# === HISTORY ===
elif page == txt('nav_hist'):
    st.header(f"📜 {txt('hist_head')}")
    
    if not st.session_state['history']:
        st.info(txt('hist_empty'))
    else:
        for item in st.session_state['history']:
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 1])
                c1.markdown(f"**{item['result']}**")
                c2.caption(item['conf'])
                c3.caption(item['time'])
                st.divider()

# === SETTINGS ===
elif page == txt('nav_set'):
    st.header(f"⚙️ {txt('set_head')}")
    st.toggle(txt('set_notif'), value=True)
    st.toggle(txt('set_dark'), value=True)
    
    st.markdown("---")
    st.caption("EcoSort Global v3.0 | Made with Streamlit & Python")