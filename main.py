import streamlit as st

st.set_page_config(page_title="UMD Simulations", layout="wide")

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


</style>
""", unsafe_allow_html=True)

# --- Header and Nav ---
st.markdown('<div class="umd-header">UNIVERSITY OF MARYLAND</div>', unsafe_allow_html=True)

st.markdown("""
<div class="navbar">
    <img src="http://math.umd.edu/~jon712/STAT400/UMD_CMNS_Math.png" alt="UMD Math Logo">
    <div class="nav-links">
        <a href="#">Home</a>
        <a href="#">Table of Contents</a>
        <a href="#">Videos &#x25BC;</a>
        <a href="#">About</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Page Title ---
st.markdown('<div class="main-title">Simulations</div>', unsafe_allow_html=True)

# 4 Image Columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("image1.png")
    st.button("Law of Larger Number")  # Will adhere to fixed CSS size
with col2:
    st.image("image2.png")  # Will adhere to fixed CSS size
    st.button("Density and Cumulative Distribution")
with col3:
    st.image("image3.png")  # Will adhere to fixed CSS size
    st.button("Calculating Probabilities")
with col4:
    st.image("image4.png")  # Will adhere to fixed CSS size
    st.button("Sampling Distribution")
    

# End Main Box
st.markdown('</div>', unsafe_allow_html=True)

