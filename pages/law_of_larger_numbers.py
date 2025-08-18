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