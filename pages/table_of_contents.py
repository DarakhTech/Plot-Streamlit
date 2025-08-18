import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="Probability and Statistics Topics", layout="wide")

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
    st.image("https://math.umd.edu/~jon712/STAT400/UMD_CMNS_Math.png", use_container_width=True)
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
html, body, [data-testid="stAppViewContainer"], .main {
    background: #e21833;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 3em;
    font-weight: bold;
    margin-top: 50px;
    color: white;
}

.topic-box {
    background-color: #f5f5f5;
    color: black;
    border-radius: 10px;
    padding: 30px;
    width: 600px;
    margin: 0 auto;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
}

.topic-box ol {
    padding-left: 20px;
}

.topic-box li {
    margin: 15px 0;
    font-size: 1.2em;
}

.topic-box a {
    color: #a60000;
    text-decoration: none;
    font-weight: bold;
}

.topic-box a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="main-title">Introduction to Probability and Statistics</div>', unsafe_allow_html=True)

# --- Topic List with Links ---
st.markdown("""
<div class="topic-box">
    <ol>
        <li><a href="https://math.umd.edu/~jon712/STAT400/asset/Chapter%201/Chapter_1_Notes/Chapter1.pdf#Probability" target="_blank">Descriptive Statistics</a></li>
        <li><a href="https://math.umd.edu/~jon712/STAT400/asset/Chapter%202/Chapter_2_Notes/Chapter2.pdf#Probability" target="_blank">Probability</a></li>
        <li><a href="https://math.umd.edu/~jon712/STAT400/asset/Chapter%203/Chapter_3_Notes/Chapter3.pdf#Random%20Variables" target="_blank">Random Variables</a></li>
        <li><a href="https://math.umd.edu/~jon712/STAT400/asset/Chapter%204/Chapter_4_Notes/Chapter4.pdf#page=2" target="_blank">Discrete Random Variables</a></li>
        <li><a href="https://math.umd.edu/~jon712/STAT400/asset/Chapter%205/Chapter_5_Notes/Chapter5.pdf#page=2" target="_blank">Continuous Random Variables</a></li>
        <li><a href="https://math.umd.edu/~jon712/STAT400/asset/Chapter%206/Chapter_6_Notes/Chapter6.pdf#page=2" target="_blank">Joint Distributions</a></li>
        <li><a href="https://math.umd.edu/~jon712/STAT400/asset/Chapter%207/Chapter_7_Notes/Chapter7.pdf#page=2" target="_blank">Random Samples and Statistics</a></li>
        <li><a href="https://math.umd.edu/~jon712/STAT400/asset/Stat400_Concepts.pdf#Point%20Estimators" target="_blank">Point Estimators</a></li>
        <li><a href="https://math.umd.edu/~jon712/STAT400/asset/Chapter%209/Chapter_9_Notes/Chapter9.pdf#Inference%20Statistic" target="_blank">Inference Statistic</a></li>
    </ol>
</div>
""", unsafe_allow_html=True)