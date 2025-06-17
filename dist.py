import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define distributions 
continuous_dists = ["Uniform Continuous", "Normal", "Gamma", "Exponential", "Pareto", "Beta Distribution"]
discrete_dists = ["Uniform", "Bernoulli", "Binomial", "Hypergeometric", "Geometric", "Negative Binomial", "Poisson"]

# Distribution handler
def get_distribution_data(dist_name, params, dist_type):
    x = pdf = cdf = mean = std = None

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
            std = np.sqrt(a * b / ((a + b)**2 * (a + b + 1)))

    else:  # Discrete
        if dist_name == "Uniform":
            a, b = params["a"], params["b"]
            x = np.arange(a, b + 1)
            pdf = [1 / (b - a + 1)] * len(x)
            cdf = np.cumsum(pdf)
            mean = (a + b) / 2
            std = np.sqrt(((b - a + 1)**2 - 1) / 12)

        elif dist_name == "Bernoulli":
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

    return x, pdf, cdf, mean, std

# Input widget rendering
def get_params(dist_name, key_prefix):
    params = {}
    if dist_name == "Normal":
        params["mu"] = st.number_input("Mean (μ)", value=0.0, key=f"{key_prefix}_mu")
        params["sigma"] = st.number_input("Std Dev (σ)", value=1.0, min_value=0.01, key=f"{key_prefix}_sigma")
    elif dist_name == "Uniform Continuous" or dist_name == "Uniform":
        params["a"] = st.number_input("Lower Bound (a)", value=0.0, key=f"{key_prefix}_a")
        params["b"] = st.number_input("Upper Bound (b)", value=1.0, key=f"{key_prefix}_b")
    elif dist_name == "Gamma":
        params["shape"] = st.number_input("Shape", value=2.0, key=f"{key_prefix}_shape")
        params["scale"] = st.number_input("Scale", value=1.0, key=f"{key_prefix}_scale")
    elif dist_name == "Exponential":
        params["rate"] = st.number_input("Rate", value=1.0, key=f"{key_prefix}_rate")
    elif dist_name == "Pareto":
        params["b"] = st.number_input("Shape (b)", value=2.0, key=f"{key_prefix}_b")
    elif dist_name == "Beta Distribution":
        params["a"] = st.number_input("Alpha (a)", value=2.0, key=f"{key_prefix}_a")
        params["b"] = st.number_input("Beta (b)", value=5.0, key=f"{key_prefix}_b")
    elif dist_name in ["Bernoulli", "Binomial", "Geometric", "Negative Binomial"]:
        if dist_name in ["Bernoulli", "Geometric"]:
            params["p"] = st.slider("p", 0.0, 1.0, 0.5, key=f"{key_prefix}_p")
        else:
            params["n"] = st.number_input("n", min_value=1, value=10, key=f"{key_prefix}_n")
            params["p"] = st.slider("p", 0.0, 1.0, 0.5, key=f"{key_prefix}_p")
    elif dist_name == "Poisson":
        params["mu"] = st.number_input("Mean (μ)", value=3.0, key=f"{key_prefix}_mu")
    return params

# --- UI Layout ---
st.title("📊 Probability Distribution Comparator")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟩 Distribution 1")
    dist_type_1 = st.radio("Type", ["Continuous", "Discrete"], key="dist_type1")
    dist1 = st.selectbox("Select Distribution", continuous_dists if dist_type_1 == "Continuous" else discrete_dists, key="dist1")
    params1 = get_params(dist1, "dist1")

with col2:
    st.markdown("### 🟧 Distribution 2")
    dist_type_2 = st.radio("Type", ["Continuous", "Discrete"], key="dist_type2")
    dist2 = st.selectbox("Select Distribution", continuous_dists if dist_type_2 == "Continuous" else discrete_dists, key="dist2")
    params2 = get_params(dist2, "dist2")

# Get data
x1, pdf1, cdf1, mean1, std1 = get_distribution_data(dist1, params1, dist_type_1)
x2, pdf2, cdf2, mean2, std2 = get_distribution_data(dist2, params2, dist_type_2)

label1 = f"{dist1} (μ={mean1:.2f}, σ={std1:.2f})" if mean1 is not None and std1 is not None else dist1
label2 = f"{dist2} (μ={mean2:.2f}, σ={std2:.2f})" if mean2 is not None and std2 is not None else dist2

# PDF/PMF Plot
st.subheader("PDF/PMF Plot")
fig_pdf, ax_pdf = plt.subplots()
if dist_type_1 == "Discrete":
    ax_pdf.bar(x1, pdf1, label=label1, alpha=0.6)
else:
    ax_pdf.plot(x1, pdf1, label=label1)

if dist_type_2 == "Discrete":
    ax_pdf.bar(x2, pdf2, label=label2, alpha=0.6, color='orange')
else:
    ax_pdf.plot(x2, pdf2, label=label2, color='orange')

ax_pdf.set_title("PDF/PMF Comparison")
ax_pdf.legend()
st.pyplot(fig_pdf)

# CDF Plot
st.subheader("CDF Plot")
fig_cdf, ax_cdf = plt.subplots()
if dist_type_1 == "Discrete":
    ax_cdf.step(x1, cdf1, where="post", label=label1)
else:
    ax_cdf.plot(x1, cdf1, label=label1)

if dist_type_2 == "Discrete":
    ax_cdf.step(x2, cdf2, where="post", label=label2, color='orange')
else:
    ax_cdf.plot(x2, cdf2, label=label2, color='orange')

ax_cdf.set_title("CDF Comparison")
ax_cdf.legend()
st.pyplot(fig_cdf)
