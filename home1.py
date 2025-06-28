import streamlit as st

# Page Config
st.set_page_config(page_title="Simulations", layout="wide")

# Inject CSS
st.markdown("""
    <style>
        /* Global and background */
        html, body, [data-testid="stAppViewContainer"], .main {
            background: linear-gradient(to bottom right, #2c003e, #540000) !important;
            color: white;
            padding: 0 10px;
        }

        /* Header styling */
        .header-container {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            background-color: #8B0000;
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 24px;
        }

        .header-left {
            display: flex;
            align-items: center;
            flex: 1;
            flex-wrap: wrap;
        }

        .header-left img {
            height: 50px;
            margin-right: 10px;
        }

        .header-title {
            font-size: 1.1rem;
            font-weight: bold;
            color: white;
        }

        .nav-links {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        .nav-links a {
            color: white;
            font-weight: bold;
            text-decoration: none;
            font-size: 0.95rem;
        }

        /* Title */
        .main-title {
            text-align: center;
            font-size: 2em;
            font-weight: bold;
            color: #FFD700;
            margin-top: 10px;
            margin-bottom: 25px;
        }

        /* Image boxes */
        .img-box {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 0 8px rgba(255, 255, 255, 0.15);
            text-align: center;
            margin-bottom: 20px;
        }

        .img-box img {
            width: 100%;
            max-width: 100%;
            height: auto;
            object-fit: contain;
            border-radius: 5px;
        }

        /* Button group */
        .button-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
            margin-bottom: 40px;
        }

        .stButton > button {
            background-color: #FFD700;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            width: 90vw;
            max-width: 300px;
        }

         @media (max-width: 768px) {
            .nav-links {
                opacity: 0 !important;
                pointer-events: none !important;
            }
        }
            .header-left {
                flex-direction: row;
                justify-content: flex-start;
                width: 100%;
            }

            .nav-links {
                width: 100%;
                justify-content: space-between;
            }

            .main-title {
                font-size: 1.6em;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Header HTML
st.markdown("""
    <div class="header-container">
        <div class="header-left">
            <img src="https://upload.wikimedia.org/wikipedia/en/thumb/7/75/University_of_Maryland_seal.svg/1200px-University_of_Maryland_seal.svg.png" alt="UMD Logo">
            <div class="header-title">DEPARTMENT OF MATHEMATICS</div>
        </div>
        <div class="nav-links">
            <a href="#">Home</a>
            <a href="#">Topics</a>
            <a href="#">Contact</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">Simulations</div>', unsafe_allow_html=True)

# Images (responsive)
st.markdown('<div class="img-box">', unsafe_allow_html=True)
st.image("image1.png", use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="img-box">', unsafe_allow_html=True)
st.image("image2.png", use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Buttons
st.markdown('<div class="button-wrapper">', unsafe_allow_html=True)
st.button("Law of Larger Number")
st.button("Density and Cumulative Distribution")
st.button("Calculating Probabilities")
st.button("Sampling Distribution")
st.markdown('</div>', unsafe_allow_html=True)
