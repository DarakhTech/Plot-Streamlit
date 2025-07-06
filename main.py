import streamlit as st

st.set_page_config(page_title="UMD Simulations", layout="wide")

# --- NAV BAR ELEMENT ---
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# Build Custom Sidebar
with st.sidebar:
    # st.image("https://math.umd.edu/~jon712/STAT400/UMD_CMNS_Math.png", use_container_width=True)
    st.header("Navigation")

    st.page_link("main.py", label="Home")
    st.page_link("pages/table_of_contents.py", label="Table of Contents")
    with st.expander("Simulations", expanded=True):
        st.page_link("pages/pdf_cdf_distribution.py", label="PDF CDF Distribution")
        st.page_link("pages/law_of_larger_numbers.py", label="Law of Large_Number")
        st.page_link("pages/calculating_prob_std_distribution.py", label="Calculating Probabilities for standard Distribution")
        st.page_link("pages/sampling_dist_cmn_stats.py", label="Sampling Distribution")
    
    st.page_link("https://www.youtube.com/playlist?list=PL90IJGPVcgidgadbkRBzMbsjeGRcDxPXR", label="Videos")
    st.page_link("https://math.umd.edu/~jon712/index.html", label="About")
# --- NAV BAR ELEMENT ---

# --- CSS Styling ---

st.markdown("""
<style>
/* Reset & page background */
html, body, [data-testid="stAppViewContainer"], .main {
    background: #e21833;
    color: black;
}

/* HEADER BAR */
.umd-header {
    background-color: #e31c3d;
    color: white;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    padding: 15px 0;
    width: 100%;
    position: absolute;
    # margin-top: -40px;
}

/* NAVBAR SECTION */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 15px 40px;
    margin-top: 70px;
    background-color: #f5f5f5;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

/* LOGO IMAGE */
.navbar img {
    height: 60px;
}

/* NAV LINKS */
.nav-links {
    display: flex;
    gap: 40px;
    font-weight: bold;
    font-size: 18px;
}

.nav-links a {
    color: #a60000;
    text-decoration: none;
}

.nav-links a:hover {
    text-decoration: underline;
            

}
            
             @media (max-width: 768px) {
            .nav-links {
                opacity: 0 !important;
                pointer-events: none !important;
            }
        }

/* Title styling */
.main-title {
    text-align: center;
            color: white;
    font-size: 3em;
    font-weight: bold;
    margin-top: 50px;       
}

            .tile-row {
    display: flex;
    flex-wrap: wrap; /* Wrap on smaller screens */
    justify-content: center;
    gap: 20px;
    margin-top: 30px;
}

.img-box {
    position: relative;
    background-color: white;
    border-radius: 10px;
    padding: 0;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
    text-align: center;
    overflow: hidden;
    width: 320px;
    height: 240px;
}

.img-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 10px;
    display: block;
    aspect-ratio: 4/3;
}

.img-box .stButton {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
}

.stButton > button {
    background-color: white;
    color: black;
    font-weight: bold;
    border-radius: 8px;
    width: 100%;
    padding: 8px;
}
    .uniform-image img {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        object-fit: cover;
        width: 100%;
        height: 200px;
    }
    .link-label {
        text-align: center;
        display: block;
        margin-top: 10px;
        font-weight: bold;
        font-size: 16px;
    }
    /* All images inside stImage elements */
    [data-testid="stImage"] img {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        object-fit: cover;
        width: 100%;
        height: 200px;
    }
    


</style>
""", unsafe_allow_html=True)

# --- Header and Nav ---
st.markdown('<div class="umd-header">UNIVERSITY OF MARYLAND</div>', unsafe_allow_html=True)

st.markdown("""
<div class="navbar">
    <img src="http://math.umd.edu/~jon712/STAT400/UMD_CMNS_Math.png" alt="UMD Math Logo">
    <div class="nav-links">
        <a href="https://math.umd.edu/~jon712/index.html">Home</a>
        <a href="#">Table of Contents</a>
        <a href="https://www.youtube.com/playlist?list=PL90IJGPVcgidgadbkRBzMbsjeGRcDxPXR">Videos</a>
        <a href="https://math.umd.edu/~jon712/index.html">About</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Page Title ---
st.markdown('<div class="main-title">Simulations</div>', unsafe_allow_html=True)

# 4 Image Columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <a href="/law_of_larger_numbers" target="_self">
            <img src="https://math.umd.edu/~jon712/STAT400/asset/trial_true.png" style="width: 100%; height: 200px; object-fit: cover; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);" />
            <div style="text-align: center; font-weight: bold; margin-top: 8px; color: white;">Law of Large Numbers</div>
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <a href="/pdf_cdf_distribution" target="_self">
            <img src="https://math.umd.edu/~jon712/STAT400/asset/Distribution.png" style="width: 100%; height: 200px; object-fit: cover; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);" />
            <div style="text-align: center; font-weight: bold; margin-top: 8px; color: white;">Density and Cumulative Distribution</div>
        </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <a href="/calculating_prob_std_distribution" target="_self">
            <img src="https://math.umd.edu/~jon712/STAT400/asset/Distribution_with_slider.png" style="width: 100%; height: 200px; object-fit: cover; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);" />
            <div style="text-align: center; font-weight: bold; margin-top: 8px; color: white;">Calculating Probabilities</div>
        </a>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <a href="/sampling_dist_cmn_stats" target="_self">
            <img src="https://math.umd.edu/~jon712/STAT400/asset/statistic_webpage.png" style="width: 100%; height: 200px; object-fit: cover; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);" />
            <div style="text-align: center; font-weight: bold; margin-top: 8px; color: white;">Sampling Distribution</div>
        </a>
    """, unsafe_allow_html=True)

# End Main Box
st.markdown('</div>', unsafe_allow_html=True)

