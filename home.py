import streamlit as st

st.set_page_config(page_title="Stimulations", layout="wide")

st.title("Main Page")
# Custom CSS
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"], .main {
            background: linear-gradient(to bottom right, #2c003e, #540000) !important;
            color: white;
        }

        .main-title {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            color: #FFD700;
            margin-top: 30px;
            margin-bottom: 50px;
        }

        .img-box {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.15);
            text-align: center;
            transition: transform 0.2s ease-in-out;
        }

        .img-box:hover {
            transform: scale(1.03);
            box-shadow: 0 0 25px rgba(255, 255, 255, 0.3);
        }

        .img-box img {
            width: 100%;
            height: 220px;
            object-fit: contain;
            border-radius: 6px;
        }

        .stButton > button {
            background-color: #FFD700;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            width: 90%;
            margin-top: 15px;
        }

        .spacer {
            margin-bottom: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">Stimulations</div>', unsafe_allow_html=True)

# Layout in 2×2 Grid
cols = st.columns(2)

cards = [
    ("image1.png", "Law of Larger Number"),
    ("image2.png", "Density and Cumulative Distribution"),
    ("image3.png", "Calculating Probabilities"),
    ("image4.png", "Sampling Distribution"),
]

for i, (img, label) in enumerate(cards):
    with cols[i % 2]:
        st.markdown('<div class="img-box">', unsafe_allow_html=True)
        st.image(img)
        st.button(label, key=label)
        st.markdown('</div><div class="spacer"></div>', unsafe_allow_html=True)
