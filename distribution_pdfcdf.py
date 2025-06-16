import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define distributions
continuous_dists = ["Uniform Continuous", "Normal", "Gamma", "Exponential", "Pareto", "Beta Distribution"]
discrete_dists = ["Uniform", "Bernoulli", "Binomial", "Hypergeometric", "Geometric", "Negative Binomial", "Poisson"]

def get_distribution_data(dist_name, params, dist_type):
    x = None
    pdf = None
    cdf = None
    mean = None
    std = None

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
            pdf = stats.uniform.pdf(x, a, b - a)
            cdf = stats.uniform.cdf(x, a, b - a)
            mean = (a + b) / 2
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
            mean = scale
            std = scale

        elif dist_name == "Pareto":
            b = params["b"]
            x = np.linspace(stats.pareto.ppf(0.01, b), stats.pareto.ppf(0.99, b), 500)
            pdf = stats.pareto.pdf(x, b)
            cdf = stats.pareto.cdf(x, b)
            if b > 1:
                mean = b / (b - 1)
            if b > 2:
                std = np.sqrt(b / ((b - 1)**2 * (b - 2)))

        elif dist_name == "Beta Distribution":
            a, b = params["a"], params["b"]
            x = np.linspace(0, 1, 500)
            pdf = stats.beta.pdf(x, a, b)
            cdf = stats.beta.cdf(x, a, b)
            mean = a / (a + b)
            std = np.sqrt(a * b / ((a + b)**2 * (a + b + 1)))

    else:  # Discrete
        if dist_name == "Bernoulli":
            p = params["p"]
            x = [0, 1]
            pdf = stats.bernoulli.pmf(x, p)
            cdf = stats.bernoulli.cdf(x, p)
            mean = p
            std = np.sqrt(p * (1 - p))

        elif dist_name == "Binomial":
            n, p = params["n"], params["p"]
            x = np.arange(0, n+1)
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

    return x, pdf, cdf, mean, std

def get_params(dist_name, key_prefix):
    params = {}
    if dist_name == "Normal":
        params["mu"] = st.number_input(f"{dist_name}: Mean (mu)", value=0.0, key=f"{key_prefix}_mu")
        params["sigma"] = st.number_input(f"{dist_name}: Std Dev (sigma)", value=1.0, min_value=0.01, key=f"{key_prefix}_sigma")
    elif dist_name == "Uniform Continuous":
        params["a"] = st.number_input(f"{dist_name}: Lower Bound (a)", value=0.0, key=f"{key_prefix}_a")
        params["b"] = st.number_input(f"{dist_name}: Upper Bound (b)", value=1.0, key=f"{key_prefix}_b")
    elif dist_name == "Gamma":
        params["shape"] = st.number_input(f"{dist_name}: Shape", value=2.0, key=f"{key_prefix}_shape")
        params["scale"] = st.number_input(f"{dist_name}: Scale", value=1.0, key=f"{key_prefix}_scale")
    elif dist_name == "Exponential":
        params["rate"] = st.number_input(f"{dist_name}: Rate", value=1.0, key=f"{key_prefix}_rate")
    elif dist_name == "Pareto":
        params["b"] = st.number_input(f"{dist_name}: Shape (b)", value=2.0, key=f"{key_prefix}_b")
    elif dist_name == "Beta Distribution":
        params["a"] = st.number_input(f"{dist_name}: Alpha (a)", value=2.0, key=f"{key_prefix}_a")
        params["b"] = st.number_input(f"{dist_name}: Beta (b)", value=5.0, key=f"{key_prefix}_b")
    elif dist_name == "Bernoulli":
        params["p"] = st.slider(f"{dist_name}: p", 0.0, 1.0, 0.5, key=f"{key_prefix}_p")
    elif dist_name == "Binomial":
        params["n"] = st.number_input(f"{dist_name}: n", min_value=1, value=10, key=f"{key_prefix}_n")
        params["p"] = st.slider(f"{dist_name}: p", 0.0, 1.0, 0.5, key=f"{key_prefix}_p")
    elif dist_name == "Poisson":
        params["mu"] = st.number_input(f"{dist_name}: Mean (mu)", value=3.0, key=f"{key_prefix}_mu")
    elif dist_name == "Geometric":
        params["p"] = st.slider(f"{dist_name}: p", 0.0, 1.0, 0.5, key=f"{key_prefix}_p")
    elif dist_name == "Negative Binomial":
        params["n"] = st.number_input(f"{dist_name}: Failures (n)", min_value=1, value=5, key=f"{key_prefix}_n")
        params["p"] = st.slider(f"{dist_name}: p", 0.0, 1.0, 0.5, key=f"{key_prefix}_p")
    return params

# Streamlit UI
st.title("Compare Two Probability Distributions")
dist_type = st.radio("Choose Distribution Type", ["Continuous", "Discrete"])

col1, col2 = st.columns(2)

with col1:
    dist1 = st.selectbox("Distribution 1", continuous_dists if dist_type == "Continuous" else discrete_dists, key="dist1")
    params1 = get_params(dist1, "dist1")

with col2:
    dist2 = st.selectbox("Distribution 2", continuous_dists if dist_type == "Continuous" else discrete_dists, key="dist2")
    params2 = get_params(dist2, "dist2")

# Get data
x1, pdf1, cdf1, mean1, std1 = get_distribution_data(dist1, params1, dist_type)
x2, pdf2, cdf2, mean2, std2 = get_distribution_data(dist2, params2, dist_type)

# Plot PDF/PMF
st.subheader("PDF/PMF Plot")
fig_pdf, ax_pdf = plt.subplots()
ax_pdf.plot(x1, pdf1, label=f"{dist1} (μ={mean1:.2f}, σ={std1:.2f})")
ax_pdf.plot(x2, pdf2, label=f"{dist2} (μ={mean2:.2f}, σ={std2:.2f})")
ax_pdf.legend()
ax_pdf.set_title("PDF/PMF Comparison")
st.pyplot(fig_pdf)

# Plot CDF
st.subheader("CDF Plot")
fig_cdf, ax_cdf = plt.subplots()
ax_cdf.plot(x1, cdf1, label=dist1)
ax_cdf.plot(x2, cdf2, label=dist2)
ax_cdf.legend()
ax_cdf.set_title("CDF Comparison")
st.pyplot(fig_cdf)