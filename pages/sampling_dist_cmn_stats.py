import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, chi2

# Set page configuration
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

# Control & Plot area side by side
with st.container():
    col_form, col_plot = st.columns([1, 3], gap="large")
    with col_form:
        with st.form("controls"):
            st.markdown("### Distribution")
            dist_type = st.selectbox("Distribution", [
                "Normal", "Uniform Continuous", "Gamma", "Exponential", "Pareto", "Beta Distribution",
                "Uniform Discrete", "Bernoulli", "Binomial", "Hypergeometric", "Geometric", "Negative Binomial", "Poisson"
            ])

            if dist_type == "Normal":
                mean = st.number_input("Mean:", value=0.0)
                std_dev = st.number_input("Standard Deviation:", value=1.0, min_value=0.01)
            elif dist_type == "Uniform Continuous":
                a = st.number_input("Min:", value=0.0)
                b = st.number_input("Max:", value=10.0)
            elif dist_type == "Gamma":
                shape = st.number_input("Shape:", value=2.0)
                scale = st.text_input("Scale (e.g., 1 or 1/3):", value="1")
            elif dist_type == "Exponential":
                rate = st.number_input("Rate (lambda):", value=1.0)
            elif dist_type == "Pareto":
                shape = st.number_input("Shape (alpha):", value=2.0)
                location = st.number_input("Location (xm):", value=1.0)
            elif dist_type == "Beta Distribution":
                alpha = st.number_input("Alpha (shape1):", value=2.0)
                beta_param = st.number_input("Beta (shape2):", value=5.0)
            elif dist_type == "Bernoulli":
                p = st.slider("Probability (p):", 0.0, 1.0, 0.5)
            elif dist_type == "Binomial":
                n_binom = st.number_input("n:", value=10)
                p = st.slider("p:", 0.0, 1.0, 0.5)
            elif dist_type == "Uniform Discrete":
                a = st.number_input("Min (int):", value=1)
                b = st.number_input("Max (int):", value=10)
            elif dist_type == "Geometric":
                p = st.slider("p:", 0.0, 1.0, 0.5)
            elif dist_type == "Negative Binomial":
                r = st.number_input("Number of failures (r):", value=10)
                p = st.slider("p:", 0.0, 1.0, 0.5)
            elif dist_type == "Hypergeometric":
                N = st.number_input("Total population (N):", value=50)
                K = st.number_input("Total success (K):", value=20)
                n = st.number_input("Sample size (n):", value=10)
            elif dist_type == "Poisson":
                lam = st.number_input("Lambda:", value=3.0)

            stat = st.selectbox("Statistic", ["min", "max", "mean", "median", "sd", "var", "sum"])
            st.markdown("### Sample Sizes")
            c1, c2 = st.columns(2)
            with c1:
                n1 = st.number_input("Sample Size 1", value=5, min_value=1)
                n3 = st.number_input("Sample Size 3", value=20, min_value=1)
            with c2:
                n2 = st.number_input("Sample Size 2", value=10, min_value=1)
                n4 = st.number_input("Sample Size 4", value=50, min_value=1)

            submit = st.form_submit_button("Submit")

    with col_plot:
        def compute_stat(x, stat):
            return {
                "min": lambda x, **kwargs: np.min(x, **kwargs),
                "max": lambda x, **kwargs: np.max(x, **kwargs),
                "mean": lambda x, **kwargs: np.mean(x, **kwargs),
                "median": lambda x, **kwargs: np.median(x, **kwargs),
                "sd": lambda x, **kwargs: np.std(x, ddof=1, **kwargs),
                "var": lambda x, **kwargs: np.var(x, ddof=1, **kwargs),
                "sum": lambda x, **kwargs: np.sum(x, **kwargs)
            }.get(stat, lambda x, **kwargs: x)(x, axis=1)

        def get_true_mean_sd(dist_type):
            try:
                if dist_type == "Normal":
                    return mean, std_dev
                elif dist_type == "Uniform Continuous":
                    mu = (a + b) / 2
                    sd = (b - a) / np.sqrt(12)
                    return mu, sd
                elif dist_type == "Gamma":
                    s = eval(scale)
                    return shape * s, np.sqrt(shape * (s ** 2))
                elif dist_type == "Exponential":
                    return 1 / rate, 1 / rate
                elif dist_type == "Pareto":
                    mu = location * shape / (shape - 1) if shape > 1 else np.nan
                    sd = np.sqrt((shape * location ** 2) / ((shape - 2) * (shape - 1) ** 2)) if shape > 2 else np.nan
                    return mu, sd
                elif dist_type == "Beta Distribution":
                    mu = alpha / (alpha + beta_param)
                    sd = np.sqrt(alpha * beta_param / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1)))
                    return mu, sd
                elif dist_type == "Bernoulli":
                    return p, np.sqrt(p * (1 - p))
                elif dist_type == "Binomial":
                    return n_binom * p, np.sqrt(n_binom * p * (1 - p))
                elif dist_type == "Uniform Discrete":
                    mu = (a + b) / 2
                    sd = np.sqrt(((b - a + 1) ** 2 - 1) / 12)
                    return mu, sd
                elif dist_type == "Geometric":
                    return 1 / p, np.sqrt((1 - p) / p ** 2)
                elif dist_type == "Negative Binomial":
                    return r * (1 - p) / p, np.sqrt(r * (1 - p) / p ** 2)
                elif dist_type == "Poisson":
                    return lam, np.sqrt(lam)
                else:
                    return 0, 1
            except:
                return np.nan, np.nan

        def calculate_bins(sample_size):
            return max(10, int(60 / np.log2(sample_size + 1)))

        if submit:
            sizes = [n1, n2, n3, n4]
            cols = st.columns(2)
            for idx, size in enumerate(sizes):
                size = int(size)
                np.random.seed(42)
                samples = None

                if dist_type == "Normal":
                    samples = np.random.normal(loc=mean, scale=std_dev, size=(10000, size))
                elif dist_type == "Uniform Continuous":
                    samples = np.random.uniform(low=a, high=b, size=(10000, size))
                elif dist_type == "Gamma":
                    samples = np.random.gamma(shape=shape, scale=eval(scale), size=(10000, size))
                elif dist_type == "Exponential":
                    samples = np.random.exponential(scale=1/rate, size=(10000, size))
                elif dist_type == "Pareto":
                    samples = (np.random.pareto(a=shape, size=(10000, size)) + 1) * location
                elif dist_type == "Beta Distribution":
                    samples = np.random.beta(a=alpha, b=beta_param, size=(10000, size))
                elif dist_type == "Bernoulli":
                    samples = np.random.binomial(n=1, p=p, size=(10000, size))
                elif dist_type == "Binomial":
                    samples = np.random.binomial(n=int(n_binom), p=p, size=(10000, size))
                elif dist_type == "Uniform Discrete":
                    samples = np.random.randint(low=int(a), high=int(b)+1, size=(10000, size))
                elif dist_type == "Geometric":
                    samples = np.random.geometric(p=p, size=(10000, size))
                elif dist_type == "Negative Binomial":
                    samples = np.random.negative_binomial(n=int(r), p=p, size=(10000, size))
                elif dist_type == "Hypergeometric":
                    samples = np.random.hypergeometric(ngood=K, nbad=N-K, nsample=n, size=(10000, size))
                elif dist_type == "Poisson":
                    samples = np.random.poisson(lam=lam, size=(10000, size))

                if samples is None:
                    st.error("Invalid distribution parameters.")
                    continue

                bins = calculate_bins(size)
                stats = compute_stat(samples, stat)
                hist_y, hist_x = np.histogram(stats, bins=bins, density=True)
                hist_x_centers = 0.5 * (hist_x[1:] + hist_x[:-1])

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hist_x_centers,
                    y=hist_y,
                    name="Histogram",
                    marker_color="royalblue",
                    opacity=0.75
                ))

                mu, sd = get_true_mean_sd(dist_type)
                if stat == "mean" and not np.isnan(mu) and not np.isnan(sd):
                    x_vals = np.linspace(min(stats), max(stats), 300)
                    y_vals = norm.pdf(x_vals, loc=mu, scale=sd/np.sqrt(size))
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Theoretical Normal", line=dict(color="red")))
                elif stat == "var" and not np.isnan(sd):
                    df = size - 1
                    x_vals = np.linspace(min(stats), max(stats), 300)
                    y_vals = chi2.pdf(x_vals * df / sd**2, df=df) * df / sd**2
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Chi-squared", line=dict(color="red")))

                fig.update_layout(
                    title=f"Sample size = {size} | Statistic = {stat}",
                    xaxis_title=stat,
                    yaxis_title="Sample Statistics Probability",
                    template="plotly_dark",
                    height=350, 
                    margin=dict(l=40, r=40, t=40, b=40)
                )

                fig.add_annotation(
                    text=f"μ ≈ {np.mean(stats):.2f}<br>σ ≈ {np.std(stats):.2f}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=1, y=1,
                    xanchor="right", yanchor="top",
                    bgcolor="black", bordercolor="white", borderwidth=1
                )

                col_index = idx % 2
                row = cols[col_index]
                with row:
                    st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False}, height=400, key=f"chart_{idx}")