import streamlit as st
import numpy as np
from scipy.stats import norm, binom
import plotly.graph_objs as go

st.set_page_config(page_title="Standard Distribution Probability", layout="wide")

# --- NAV BAR ELEMENT ---
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {
    display: none;
}
.main, .block-container {
    height: 100vh !important;
    min-height: 100vh !important;
}

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .element-container {
        padding: 0rem !important;
        margin: 0rem !important;
    }
    .stPlotlyChart {
        padding: 0rem !important;
        margin: 0rem !important;
    }
    
</style>
"""


st.markdown(hide_sidebar, unsafe_allow_html=True)

color1 = 'yellow'
color2 = 'blue'

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


st.title("Distribution Explorer")

col1, col2 = st.columns([1, 3], gap="medium")
# Distribution type

with col1:
    st.subheader("Inputs")
    dist_type = st.selectbox("Select Distribution Type", ["Continuous", "Discrete"], key="dist_type_select")
    if dist_type == "Continuous":
        dist_name = st.selectbox("Select Continuous Distribution", [
            "Normal", "Uniform", "Gamma", "Exponential", "Pareto", "Beta"
        ], key="cont_dist")
        if dist_name == "Normal":
            mean = st.number_input("Mean", value=0.0, key="mean")
            sd = st.number_input("Standard Deviation", min_value=0.0001, value=1.0, key="sd")
            xmin = mean - 10 * sd
            xmax = mean + 10 * sd
            x = np.linspace(xmin, xmax, 500)
            y = norm.pdf(x, loc=mean, scale=sd)
        elif dist_name == "Uniform":
            a = st.number_input("Lower Bound (a)", value=0.0, key="uniform_a")
            b = st.number_input("Upper Bound (b)", value=1.0, key="uniform_b")
            if b <= a:
                st.warning("Upper bound must be greater than lower bound.")
            x = np.linspace(a, b, 500)
            from scipy.stats import uniform
            y = uniform.pdf(x, loc=a, scale=b-a)
        elif dist_name == "Gamma":
            shape = st.number_input("Shape (k)", min_value=0.01, value=2.0, key="gamma_shape")
            scale = st.number_input("Scale (θ)", min_value=0.01, value=2.0, key="gamma_scale")
            from scipy.stats import gamma
            xmin = 0
            xmax = gamma.ppf(0.999, shape, scale=scale)
            x = np.linspace(xmin, xmax, 500)
            y = gamma.pdf(x, a=shape, scale=scale)
        elif dist_name == "Exponential":
            lambd = st.number_input("Rate (λ)", min_value=0.0001, value=1.0, key="exp_lambda")
            from scipy.stats import expon
            xmin = 0
            xmax = expon.ppf(0.999, scale=1/lambd)
            x = np.linspace(xmin, xmax, 500)
            y = expon.pdf(x, scale=1/lambd)
        elif dist_name == "Pareto":
            b = st.number_input("Shape (b)", min_value=0.01, value=2.0, key="pareto_b")
            from scipy.stats import pareto
            xmin = pareto.ppf(0.001, b)
            xmax = pareto.ppf(0.999, b)
            x = np.linspace(xmin, xmax, 500)
            y = pareto.pdf(x, b)
        elif dist_name == "Beta":
            a = st.number_input("Alpha (a)", min_value=0.01, value=2.0, key="beta_a")
            b = st.number_input("Beta (b)", min_value=0.01, value=2.0, key="beta_b")
            from scipy.stats import beta
            x = np.linspace(0, 1, 500)
            y = beta.pdf(x, a, b)
    else:
        dist_name = st.selectbox("Select Discrete Distribution", [
            "Binomial", "Uniform", "Bernoulli", "Hypergeometric", "Geometric", "Negative Binomial", "Poisson"
        ], key="disc_dist")
        if dist_name == "Binomial":
            n = st.number_input("Number of trials (n)", min_value=1, step=1, value=10, key="n")
            p = st.number_input("Probability of success (p)", min_value=0.0, max_value=1.0, value=0.5, key="p")
            x = np.arange(0, n + 1)
            y = binom.pmf(x, n, p)
        elif dist_name == "Uniform":
            a = st.number_input("Lower Bound (a)", min_value=0, step=1, value=0, key="dunif_a")
            b = st.number_input("Upper Bound (b)", min_value=0, step=1, value=5, key="dunif_b")
            if b < a:
                st.warning("Upper bound must be greater than or equal to lower bound.")
            from scipy.stats import randint
            x = np.arange(a, b + 1)
            y = randint.pmf(x, a, b + 1)
        elif dist_name == "Bernoulli":
            p = st.number_input("Probability of success (p)", min_value=0.0, max_value=1.0, value=0.5, key="bernoulli_p")
            from scipy.stats import bernoulli
            x = np.array([0, 1])
            y = bernoulli.pmf(x, p)
        elif dist_name == "Hypergeometric":
            M = st.number_input("N (Population Size)", min_value=1, step=1, value=20, key="hypergeo_M")
            n = st.number_input("M(Number of Successes)", min_value=0, step=1, value=7, key="hypergeo_n")
            N = st.number_input("n (Sample Size)", min_value=1, step=1, value=12, key="hypergeo_N")
            from scipy.stats import hypergeom
            x = np.arange(max(0, N + n - M), min(n, N) + 1)
            y = hypergeom.pmf(x, M, n, N)
        elif dist_name == "Geometric":
            p = st.number_input("Probability of success (p)", min_value=0.0001, max_value=1.0, value=0.5, key="geom_p")
            from scipy.stats import geom
            x = np.arange(1, 16)
            y = geom.pmf(x, p)
        elif dist_name == "Negative Binomial":
            r = st.number_input("Number of successes (r)", min_value=1, step=1, value=5, key="nbinom_r")
            p = st.number_input("Probability of success (p)", min_value=0.0001, max_value=1.0, value=0.5, key="nbinom_p")
            from scipy.stats import nbinom
            x = np.arange(0, 30)
            y = nbinom.pmf(x, r, p)
        elif dist_name == "Poisson":
            mu = st.number_input("Mean (mu)", min_value=0.0001, value=3.0, key="poisson_mu")
            from scipy.stats import poisson
            x = np.arange(0, 20)
            y = poisson.pmf(x, mu)

    # --- Independent slider and number inputs, no interlinking ---
    # --- Slider with visible plus/minus buttons for fine-tuning (no number input) ---
    st.markdown("**Select Range**")
    if dist_type == "Continuous" and dist_name in ["Normal", "Uniform", "Gamma", "Exponential", "Pareto", "Beta"]:
        min_val = float(x[0])
        max_val = float(x[-1])
        step = 0.01
        val_type = float
    else:
        min_val = int(x[0])
        max_val = int(x[-1])
        step = 1
        val_type = int

    # Use session state to store slider values for button nudging
    if "slider_lower" not in st.session_state:
        st.session_state["slider_lower"] = val_type(min_val)
    if "slider_upper" not in st.session_state:
        st.session_state["slider_upper"] = val_type(max_val)

    col_l, col_slider, col_u = st.columns([1, 8, 1])
    with col_l:
        if st.button("➖", key="lower_minus"):
            st.session_state["slider_lower"] = val_type(max(min_val, st.session_state["slider_lower"] - step))
        if st.button("＋", key="lower_plus"):
            st.session_state["slider_lower"] = val_type(min(st.session_state["slider_upper"], st.session_state["slider_lower"] + step))
    with col_u:
        if st.button("➖", key="upper_minus"):
            st.session_state["slider_upper"] = val_type(max(st.session_state["slider_lower"], st.session_state["slider_upper"] - step))
        if st.button("＋", key="upper_plus"):
            st.session_state["slider_upper"] = val_type(min(max_val, st.session_state["slider_upper"] + step))
    with col_slider:
        slider_vals = st.slider(
            "Range", min_value=val_type(min_val), max_value=val_type(max_val),
            value=(val_type(st.session_state["slider_lower"]), val_type(st.session_state["slider_upper"])),
            step=val_type(step), key="range_slider")
        st.session_state["slider_lower"], st.session_state["slider_upper"] = slider_vals

    range_vals = (st.session_state["slider_lower"], st.session_state["slider_upper"])
    if range_vals[0] > range_vals[1]:
        st.warning("Lower bound cannot be greater than upper bound.")

    submit = st.button("Submit")


with col2:
    if ('submit' in locals() and submit ) or True:
        # st.subheader("PDF / PMF Plot")
        fig = go.Figure()
        if dist_type == "Continuous":
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='PDF'))
            x_fill = np.linspace(range_vals[0], range_vals[1], 300)
            if dist_name == "Normal":
                y_fill = norm.pdf(x_fill, loc=mean, scale=sd)
                prob_in_range = norm.cdf(range_vals[1], mean, sd) - norm.cdf(range_vals[0], mean, sd)
            elif dist_name == "Uniform":
                from scipy.stats import uniform
                y_fill = uniform.pdf(x_fill, loc=a, scale=b-a)
                prob_in_range = uniform.cdf(range_vals[1], loc=a, scale=b-a) - uniform.cdf(range_vals[0], loc=a, scale=b-a)
            elif dist_name == "Gamma":
                from scipy.stats import gamma
                y_fill = gamma.pdf(x_fill, a=shape, scale=scale)
                prob_in_range = gamma.cdf(range_vals[1], a=shape, scale=scale) - gamma.cdf(range_vals[0], a=shape, scale=scale)
            elif dist_name == "Exponential":
                from scipy.stats import expon
                y_fill = expon.pdf(x_fill, scale=1/lambd)
                prob_in_range = expon.cdf(range_vals[1], scale=1/lambd) - expon.cdf(range_vals[0], scale=1/lambd)
            elif dist_name == "Pareto":
                from scipy.stats import pareto
                y_fill = pareto.pdf(x_fill, b)
                prob_in_range = pareto.cdf(range_vals[1], b) - pareto.cdf(range_vals[0], b)
            elif dist_name == "Beta":
                from scipy.stats import beta
                y_fill = beta.pdf(x_fill, a, b)
                prob_in_range = beta.cdf(range_vals[1], a, b) - beta.cdf(range_vals[0], a, b)
            else:
                y_fill = np.zeros_like(x_fill)
                prob_in_range = 0.0
            fig.add_trace(go.Scatter(x=np.concatenate([[range_vals[0]], x_fill, [range_vals[1]]]),
                                     y=np.concatenate([[0], y_fill, [0]]),
                                     fill='toself',
                                     name='In Range',
                                     fillcolor=color1,
                                     opacity=0.5))
        else:
            k_min = int(np.floor(range_vals[0]))
            k_max = int(np.floor(range_vals[1]))
            colors = [color1 if (k_min <= xi <= k_max) else color2 for xi in x]
            fig.add_trace(go.Bar(x=x, y=y, marker_color=colors))
            if dist_name == "Binomial":
                prob_in_range = binom.cdf(k_max, n, p) - binom.cdf(k_min - 1, n, p)
            elif dist_name == "Uniform":
                from scipy.stats import randint
                prob_in_range = randint.cdf(k_max, a, b + 1) - randint.cdf(k_min - 1, a, b + 1)
            elif dist_name == "Bernoulli":
                from scipy.stats import bernoulli
                prob_in_range = bernoulli.cdf(k_max, p) - bernoulli.cdf(k_min - 1, p)
            elif dist_name == "Hypergeometric":
                from scipy.stats import hypergeom
                prob_in_range = hypergeom.cdf(k_max, M, n, N) - hypergeom.cdf(k_min - 1, M, n, N)
            elif dist_name == "Geometric":
                from scipy.stats import geom
                prob_in_range = geom.cdf(k_max, p) - geom.cdf(k_min - 1, p)
            elif dist_name == "Negative Binomial":
                from scipy.stats import nbinom
                prob_in_range = nbinom.cdf(k_max, r, p) - nbinom.cdf(k_min - 1, r, p)
            elif dist_name == "Poisson":
                from scipy.stats import poisson
                prob_in_range = poisson.cdf(k_max, mu) - poisson.cdf(k_min - 1, mu)
            else:
                prob_in_range = 0.0
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # st.subheader("Probability Split")
        st.write(f"Probability: in range: **{prob_in_range:.4f}** |  out of range: **{1 - prob_in_range:.4f}**")

        col21, col22 = st.columns([1.5, 1], gap='large')
        with col21:
            bar_fig = go.Figure(data=[
                go.Bar(x=['In Range', 'Out of Range'], y=[prob_in_range, 1 - prob_in_range],
                    marker_color=[color1, color2])
            ])
            bar_fig.update_layout(yaxis=dict(title="Probability"), height=300)
            st.plotly_chart(bar_fig, use_container_width=True)

        with col22:
            pie_fig = go.Figure(data=[
                go.Pie(labels=['In Range', 'Out of Range'],
                    values=[prob_in_range, 1 - prob_in_range],
                    marker_colors=[color1, color2])
            ])
            st.plotly_chart(pie_fig, use_container_width=True)