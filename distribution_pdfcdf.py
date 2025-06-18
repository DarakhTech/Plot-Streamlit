import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# -- Page Config --
st.set_page_config(page_title="Distributions", layout="wide")

# -- Custom CSS for full-screen layout --
st.markdown("""
<style>
    html, body, [data-testid="stApp"] {
        height: 100vh;
    }

    .main {
        display: flex;
        flex-direction: column;
        height: 100vh;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    .center-column {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .left-col, .right-col {
        background-color: #f9f9f9;
        border: 1px solid #ccc;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -- Distribution Lists --
continuous_dists = ["Normal", "Uniform Continuous", "Gamma", "Exponential", "Pareto", "Beta Distribution"]
discrete_dists = ["Bernoulli", "Binomial", "Poisson", "Geometric", "Negative Binomial", "Uniform"]

# -- Distribution Logic --
def get_distribution_data(dist_name, params, dist_type):
    x, pdf, cdf, mean, std = None, None, None, None, None
    if dist_type == "Continuous":
        if dist_name == "Normal":
            mu, sigma = params["mu"], params["sigma"]
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
            pdf = stats.norm.pdf(x, mu, sigma)
            cdf = stats.norm.cdf(x, mu, sigma)
            mean, std = mu, sigma
        elif dist_name == "Uniform Continuous":
            a, b = params["a"], params["b"]
            x = np.linspace(a, b, 500)
            pdf = stats.uniform.pdf(x, a, b-a)
            cdf = stats.uniform.cdf(x, a, b-a)
            mean = (a + b)/2
            std = np.sqrt((b - a)**2 / 12)
        elif dist_name == "Gamma":
            shape, scale = params["shape"], params["scale"]
            x = np.linspace(0, stats.gamma.ppf(0.99, shape, scale=scale), 500)
            pdf = stats.gamma.pdf(x, shape, scale=scale)
            cdf = stats.gamma.cdf(x, shape, scale=scale)
            mean = shape * scale
            std = np.sqrt(shape) * scale
        elif dist_name == "Exponential":
            rate = params["rate"]
            scale = 1 / rate
            x = np.linspace(0, stats.expon.ppf(0.99, scale=scale), 500)
            pdf = stats.expon.pdf(x, scale=scale)
            cdf = stats.expon.cdf(x, scale=scale)
            mean = std = scale
        elif dist_name == "Pareto":
            b = params["b"]
            x = np.linspace(stats.pareto.ppf(0.01, b), stats.pareto.ppf(0.99, b), 500)
            pdf = stats.pareto.pdf(x, b)
            cdf = stats.pareto.cdf(x, b)
            mean = b / (b - 1) if b > 1 else None
            std = np.sqrt(b / ((b - 1)**2 * (b - 2))) if b > 2 else None
        elif dist_name == "Beta Distribution":
            a, b = params["a"], params["b"]
            x = np.linspace(0, 1, 500)
            pdf = stats.beta.pdf(x, a, b)
            cdf = stats.beta.cdf(x, a, b)
            mean = a / (a + b)
            std = np.sqrt(a*b / ((a + b)**2 * (a + b + 1)))
    else:
        if dist_name == "Bernoulli":
            p = params["p"]
            x = [0, 1]
            pdf = stats.bernoulli.pmf(x, p)
            cdf = stats.bernoulli.cdf(x, p)
            mean = p
            std = np.sqrt(p * (1 - p))
        elif dist_name == "Binomial":
            n, p = params["n"], params["p"]
            x = np.arange(0, n + 1)
            pdf = stats.binom.pmf(x, n, p)
            cdf = stats.binom.cdf(x, n, p)
            mean = n * p
            std = np.sqrt(n * p * (1 - p))
        elif dist_name == "Poisson":
            mu = params["mu"]
            x = np.arange(0, 20)
            pdf = stats.poisson.pmf(x, mu)
            cdf = stats.poisson.cdf(x, mu)
            mean = mu
            std = np.sqrt(mu)
        elif dist_name == "Geometric":
            p = params["p"]
            x = np.arange(1, 15)
            pdf = stats.geom.pmf(x, p)
            cdf = stats.geom.cdf(x, p)
            mean = 1 / p
            std = np.sqrt((1 - p) / p**2)
        elif dist_name == "Negative Binomial":
            n, p = params["n"], params["p"]
            x = np.arange(0, 20)
            pdf = stats.nbinom.pmf(x, n, p)
            cdf = stats.nbinom.cdf(x, n, p)
            mean = n * (1 - p) / p
            std = np.sqrt(n * (1 - p)) / p
        elif dist_name == "Uniform":
            a, b = params["a"], params["b"]
            x = np.arange(a, b + 1)
            pdf = [1 / (b - a + 1)] * len(x)
            cdf = np.cumsum(pdf)
            mean = (a + b) / 2
            std = np.sqrt(((b - a + 1)**2 - 1) / 12)
    return x, pdf, cdf, mean, std

# -- Param Inputs --
def get_params(dist, prefix):
    p = {}
    if dist == "Normal":
        p["mu"] = st.number_input("Mean (μ):", 0.0, key=f"{prefix}_mu")
        p["sigma"] = st.number_input("Std Dev (σ):", value=1.0, min_value=0.01, key=f"{prefix}_sd")

    elif dist == "Uniform Continuous" or dist == "Uniform":
        p["a"] = st.number_input("a (Lower):", 0.0, key=f"{prefix}_a")
        p["b"] = st.number_input("b (Upper):", 10.0, key=f"{prefix}_b")
    elif dist == "Gamma":
        p["shape"] = st.number_input("Shape (α):", 2.0, key=f"{prefix}_alpha")
        p["scale"] = st.number_input("Scale (θ):", 1.0, key=f"{prefix}_scale")
    elif dist == "Exponential":
        p["rate"] = st.number_input("Rate (λ):", 1.0, key=f"{prefix}_rate")
    elif dist == "Pareto":
        p["b"] = st.number_input("Shape (b):", 2.0, key=f"{prefix}_b")
    elif dist == "Beta Distribution":
        p["a"] = st.number_input("Alpha (a):", 2.0, key=f"{prefix}_a")
        p["b"] = st.number_input("Beta (b):", 5.0, key=f"{prefix}_b")
    elif dist == "Bernoulli":
        p["p"] = st.number_input("p", min_value=0.0, max_value=1.0, value=0.5, key=f"{prefix}_p")
    elif dist == "Binomial":
        p["n"] = st.number_input("n (Trials):", 10, step=1, key=f"{prefix}_n")
        p["p"] = st.number_input("p", min_value=0.0, max_value=1.0, value=0.5, key=f"{prefix}_p")

    elif dist == "Poisson":
        p["mu"] = st.number_input("Mean (μ):", 5.0, key=f"{prefix}_mu")
    elif dist == "Geometric":
        p["p"] = st.number_input("p", min_value=0.0, max_value=1.0, value=0.5, key=f"{prefix}_p")
    elif dist == "Negative Binomial":
        p["n"] = st.number_input("Failures (n):", 5, step=1, key=f"{prefix}_n")
        p["p"] = st.number_input("p", min_value=0.0, max_value=1.0, value=0.5, key=f"{prefix}_p")
    return p

# -- App Layout --
st.title("Probability Distribution Visualisation")

cols = st.columns([3, 6, 3])  # left input, center plot, right input

# Left Column (Dist 1)
with cols[0]:
    st.subheader("Distribution 1")
    dist_type_1 = st.selectbox("Type of Distribution 1", ["Continuous", "Discrete"], key="type1")
    dist1 = st.selectbox("Choose Distribution to Display", continuous_dists if dist_type_1 == "Continuous" else discrete_dists, key="dist1")
    params1 = get_params(dist1, "p1")

# Right Column (Dist 2)
with cols[2]:
    st.subheader("Distribution 2")
    dist_type_2 = st.selectbox("Type of Distribution 2", ["Continuous", "Discrete"], key="type2")
    dist2 = st.selectbox("Choose Distribution to Display", continuous_dists if dist_type_2 == "Continuous" else discrete_dists, key="dist2")
    params2 = get_params(dist2, "p2")

# Center Column (Plots)
with cols[1]:
    x1, pdf1, cdf1, mean1, std1 = get_distribution_data(dist1, params1, dist_type_1)
    x2, pdf2, cdf2, mean2, std2 = get_distribution_data(dist2, params2, dist_type_2)

    label1 = f"{dist1}"
    label2 = f"{dist2}"

    # Determine shared x-range
    xmin = min(np.min(x1), np.min(x2))
    xmax = max(np.max(x1), np.max(x2))
    x_common = np.linspace(xmin, xmax, 500)

    st.subheader("PDF/PMF of Distributions")
    fig_pdf, ax_pdf = plt.subplots()
    ax_pdf.set_xlim(xmin, xmax)
    ax_pdf.grid(True)

    if dist_type_1 == "Discrete":
        ax_pdf.bar(x1, pdf1, label=label1, alpha=0.6,)
    else:
        pdf1_interp = np.interp(x_common, x1, pdf1)
        ax_pdf.plot(x_common, pdf1_interp, label=label1)

    if dist_type_2 == "Discrete":
        ax_pdf.bar(x2, pdf2, label=label2, alpha=0.6, color="orange")
    else:
        pdf2_interp = np.interp(x_common, x2, pdf2)
        ax_pdf.plot(x_common, pdf2_interp, label=label2, color="orange")

    ax_pdf.set_xlabel("Value")
    ax_pdf.set_ylabel("Density")
    ax_pdf.set_title("PDF/PMF Comparison")
    ax_pdf.legend()
    st.pyplot(fig_pdf)

    st.subheader("CDF of Distributions")
    fig_cdf, ax_cdf = plt.subplots()
    ax_cdf.set_xlim(xmin, xmax)
    ax_cdf.grid(True)

    if dist_type_1 == "Discrete":
        ax_cdf.step(x1, cdf1, where='post', label=label1)
    else:
        cdf1_interp = np.interp(x_common, x1, cdf1)
        ax_cdf.plot(x_common, cdf1_interp, label=label1)

    if dist_type_2 == "Discrete":
        ax_cdf.step(x2, cdf2, where='post', label=label2, color="orange")
    else:
        cdf2_interp = np.interp(x_common, x2, cdf2)
        ax_cdf.plot(x_common, cdf2_interp, label=label2, color="orange")

    ax_cdf.set_xlabel("Value")
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_title("CDF Comparison")
    ax_cdf.legend()
    st.pyplot(fig_cdf)
