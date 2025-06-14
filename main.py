import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    norm, binom, bernoulli, poisson, geom, nbinom, hypergeom,
    gamma, expon, pareto, beta, uniform
)

st.set_page_config(layout="wide")
st.title("📊 Distributions")

# Layout in columns
col1, col2 = st.columns([1, 1])

with col1:
    dist_type = st.selectbox("Type of Distribution", ["Continuous", "Discrete"])

    if dist_type == "Continuous":
        cont_options = ["Uniform Continuous", "Normal", "Gamma", "Exponential", "Pareto", "Beta Distribution"]
        dist_choice = st.selectbox("Distribution Want to Display", cont_options)
        mean = st.number_input("Mean", value=0.0)
        std = st.number_input("Standard Deviation", value=1.0)
        second_dist = st.checkbox("Second Distribution", value=False)

        if second_dist:
            second_dist_choice = st.selectbox("Second Distribution", cont_options)
            mean2 = st.number_input("Mean (2nd)", value=1.0, key="mean2")
            std2 = st.number_input("Std Dev (2nd)", value=1.0, key="std2")

    elif dist_type == "Discrete":
        disc_options = ["Bernoulli", "Binomial", "Hypergeometric", "Geometric", "Negative Binomial", "Poisson"]
        dist_choice = st.selectbox("Distribution Want to Display", disc_options)
        p = st.number_input("p", min_value=0.0, max_value=1.0, value=0.5)
        n = st.number_input("n", min_value=1, value=10)
        second_dist = st.checkbox("Second Distribution", value=False)

        if second_dist:
            second_dist_choice = st.selectbox("Second Distribution", disc_options)
            p2 = st.number_input("p (2nd)", min_value=0.0, max_value=1.0, value=0.5, key="p2")
            n2 = st.number_input("n (2nd)", min_value=1, value=10, key="n2")

    submit = st.button("Submit")

with col2:
    if submit:
        x = np.linspace(-5, 15, 1000)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        if dist_type == "Continuous":
            if dist_choice == "Normal":
                y = norm.pdf(x, loc=mean, scale=std)
                cdf = norm.cdf(x, loc=mean, scale=std)
                label = "Normal"
            elif dist_choice == "Uniform Continuous":
                y = uniform.pdf(x, loc=mean - std, scale=2 * std)
                cdf = uniform.cdf(x, loc=mean - std, scale=2 * std)
                label = "Uniform"
            elif dist_choice == "Gamma":
                y = gamma.pdf(x, a=2)
                cdf = gamma.cdf(x, a=2)
                label = "Gamma"
            elif dist_choice == "Exponential":
                y = expon.pdf(x)
                cdf = expon.cdf(x)
                label = "Exponential"
            elif dist_choice == "Pareto":
                y = pareto.pdf(x, b=2)
                cdf = pareto.cdf(x, b=2)
                label = "Pareto"
            elif dist_choice == "Beta Distribution":
                x = np.linspace(0, 1, 1000)
                y = beta.pdf(x, a=2, b=5)
                cdf = beta.cdf(x, a=2, b=5)
                label = "Beta"

            ax1.plot(x, y, label=label.lower(), color="red")
            ax2.plot(x, cdf, label=label.lower(), color="blue")

            if second_dist:
                if second_dist_choice == "Normal":
                    y2 = norm.pdf(x, loc=mean2, scale=std2)
                    cdf2 = norm.cdf(x, loc=mean2, scale=std2)
                    label2 = "Normal (2nd)"
                elif second_dist_choice == "Uniform Continuous":
                    y2 = uniform.pdf(x, loc=mean2 - std2, scale=2 * std2)
                    cdf2 = uniform.cdf(x, loc=mean2 - std2, scale=2 * std2)
                    label2 = "Uniform (2nd)"
                elif second_dist_choice == "Gamma":
                    y2 = gamma.pdf(x, a=2)
                    cdf2 = gamma.cdf(x, a=2)
                    label2 = "Gamma (2nd)"
                elif second_dist_choice == "Exponential":
                    y2 = expon.pdf(x)
                    cdf2 = expon.cdf(x)
                    label2 = "Exponential (2nd)"
                elif second_dist_choice == "Pareto":
                    y2 = pareto.pdf(x, b=2)
                    cdf2 = pareto.cdf(x, b=2)
                    label2 = "Pareto (2nd)"
                elif second_dist_choice == "Beta Distribution":
                    x2 = np.linspace(0, 1, 1000)
                    y2 = beta.pdf(x2, a=2, b=5)
                    cdf2 = beta.cdf(x2, a=2, b=5)
                    ax1.plot(x2, y2, label="beta (2nd)", linestyle="--")
                    ax2.plot(x2, cdf2, label="beta (2nd)", linestyle="--")
                    label2 = None

                if label2:
                    ax1.plot(x, y2, label=label2.lower(), linestyle="--")
                    ax2.plot(x, cdf2, label=label2.lower(), linestyle="--")

        elif dist_type == "Discrete":
            k = np.arange(0, n + 1)
            if dist_choice == "Binomial":
                y = binom.pmf(k, n, p)
                cdf = binom.cdf(k, n, p)
            elif dist_choice == "Bernoulli":
                k = [0, 1]
                y = bernoulli.pmf(k, p)
                cdf = bernoulli.cdf(k, p)
            elif dist_choice == "Poisson":
                k = np.arange(0, n + 1)
                y = poisson.pmf(k, mu=n*p)
                cdf = poisson.cdf(k, mu=n*p)
            elif dist_choice == "Geometric":
                y = geom.pmf(k, p)
                cdf = geom.cdf(k, p)
            elif dist_choice == "Negative Binomial":
                y = nbinom.pmf(k, n, p)
                cdf = nbinom.cdf(k, n, p)
            elif dist_choice == "Hypergeometric":
                M = 20
                N = 7
                y = hypergeom.pmf(k, M, N, n)
                cdf = hypergeom.cdf(k, M, N, n)

            ax1.bar(k, y, alpha=0.6, color="orange", label=dist_choice.lower())
            ax2.plot(k, cdf, "ro-", label=dist_choice.lower())

            if second_dist:
                k2 = np.arange(0, n2 + 1)
                if second_dist_choice == "Binomial":
                    y2 = binom.pmf(k2, n2, p2)
                    cdf2 = binom.cdf(k2, n2, p2)
                elif second_dist_choice == "Bernoulli":
                    k2 = [0, 1]
                    y2 = bernoulli.pmf(k2, p2)
                    cdf2 = bernoulli.cdf(k2, p2)
                elif second_dist_choice == "Poisson":
                    y2 = poisson.pmf(k2, mu=n2 * p2)
                    cdf2 = poisson.cdf(k2, mu=n2 * p2)
                elif second_dist_choice == "Geometric":
                    y2 = geom.pmf(k2, p2)
                    cdf2 = geom.cdf(k2, p2)
                elif second_dist_choice == "Negative Binomial":
                    y2 = nbinom.pmf(k2, n2, p2)
                    cdf2 = nbinom.cdf(k2, n2, p2)
                elif second_dist_choice == "Hypergeometric":
                    M = 20
                    N = 7
                    y2 = hypergeom.pmf(k2, M, N, n2)
                    cdf2 = hypergeom.cdf(k2, M, N, n2)

                ax1.bar(k2, y2, alpha=0.3, label=second_dist_choice.lower() + " (2nd)")
                ax2.plot(k2, cdf2, "go--", label=second_dist_choice.lower() + " (2nd)")

        ax1.set_title("PDF/PMF of Distributions", fontsize=13)
        ax2.set_title("CDF of Distributions", fontsize=13)
        ax1.set_xlabel("Value")
        ax2.set_xlabel("Value")
        ax1.set_ylabel("Density")
        ax2.set_ylabel("Cumulative")
        ax1.legend()
        ax2.legend()
        st.pyplot(fig)
