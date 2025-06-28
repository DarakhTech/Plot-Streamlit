import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objs as go

# -- Page Config --
st.set_page_config(
    page_title="Distributions", 
    page_icon="📊",
    layout="wide"
)

# -- Custom CSS for full-screen layout --
st.markdown("""
            
<meta name="description" content="Visualize and compare probability distributions like Normal, Poisson, and more using interactive graphs. Built with Streamlit.">
<meta property="og:title" content="Probability Distribution Visualisation">
<meta property="og:description" content="Interactive app to compare PDF and CDF of various distributions.">
<meta property="og:image" content="https://brand.umd.edu/assets/images/favicon.ico">
<meta property="og:type" content="website">
<meta property="og:url" content="https://simulations-distribution.streamlit.app">

<style>
    html, body, [data-testid="stApp"] {
        height: 100vh;
    }

    .main {
        display: flex;
        flex-direction: column;
        height: 100vh;
        padding-left: 1rem;
        padding-right: 2rem;
    }

    /* Removed sidebar hiding to allow toggling with hamburger icon */
    /* section[data-testid="stSidebar"] {
        display: none;
    } */

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }

    .center-column {
        display: flex;
        flex-direction: column;
        align-items: center;
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
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 2000)
            pdf = stats.norm.pdf(x, mu, sigma)
            cdf = stats.norm.cdf(x, mu, sigma)
            mean, std = mu, sigma
        elif dist_name == "Uniform Continuous":
            a, b = params["a"], params["b"]
            x = np.linspace(a, b, 2000)
            pdf = stats.uniform.pdf(x, a, b-a)
            cdf = stats.uniform.cdf(x, a, b-a)
            mean = (a + b)/2
            std = np.sqrt((b - a)**2 / 12)
        elif dist_name == "Gamma":
            shape, scale = params["shape"], params["scale"]
            x = np.linspace(0, stats.gamma.ppf(0.99, shape, scale=scale), 2000)
            pdf = stats.gamma.pdf(x, shape, scale=scale)
            cdf = stats.gamma.cdf(x, shape, scale=scale)
            mean = shape * scale
            std = np.sqrt(shape) * scale
        elif dist_name == "Exponential":
            rate = params["rate"]
            scale = 1 / rate
            x = np.linspace(0, stats.expon.ppf(0.99, scale=scale), 2000)
            pdf = stats.expon.pdf(x, scale=scale)
            cdf = stats.expon.cdf(x, scale=scale)
            mean = std = scale
        elif dist_name == "Pareto":
            alpha = params["alpha"]
            beta = params["beta"]
            x = np.linspace(beta, beta * 10, 2000)
            pdf = stats.pareto.pdf(x, b=alpha, scale=beta)
            cdf = stats.pareto.cdf(x, b=alpha, scale=beta)
            mean = (alpha * beta) / (alpha - 1) if alpha > 1 else None
            std = np.sqrt((alpha * beta**2) / ((alpha - 1)**2 * (alpha - 2))) if alpha > 2 else None
        elif dist_name == "Beta Distribution":
            a, b = params["a"], params["b"]
            x = np.linspace(0, 1, 2000)
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
        p["a"] = st.number_input("a (Lower):", 1.0, step=1.0, key=f"{prefix}_a")
        p["b"] = st.number_input("b (Upper):", 2.0, step=1.0, key=f"{prefix}_b")
    elif dist == "Gamma":
        p["shape"] = st.number_input("Shape (α):", 2.0, key=f"{prefix}_alpha")
        p["scale"] = st.number_input("Scale (θ):", 1.0, key=f"{prefix}_scale")
    elif dist == "Exponential":
        p["rate"] = st.number_input("Rate (λ):", 1.0, key=f"{prefix}_rate")
    elif dist == "Pareto":
        p["alpha"] = st.number_input("Alpha (α):", value=2.0, min_value=0.01, step=1.0, key=f"{prefix}_alpha")
        p["beta"] = st.number_input("Beta (β):", value=1.0, min_value=0.01, step=1.0, key=f"{prefix}_beta")
    elif dist == "Beta Distribution":
        p["a"] = st.number_input("Alpha (a):", 2.0, key=f"{prefix}_a")
        p["b"] = st.number_input("Beta (b):", 5.0, key=f"{prefix}_b")
    elif dist in ["Bernoulli", "Geometric"]:
        p["p"] = st.number_input("p", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"{prefix}_p")
    elif dist == "Binomial" or dist == "Negative Binomial":
        p["n"] = st.number_input("n (Trials):", 10, step=1, key=f"{prefix}_n")
        p["p"] = st.number_input("p", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"{prefix}_p")
    elif dist == "Poisson":
        p["mu"] = st.number_input("Mean (μ):", 5.0, key=f"{prefix}_mu")
    return p


# For CDF Comparison
def plot_distributions(x1, y1, x2, y2, label1, label2, x_label, y_label, title, dist_type_1, dist_type_2, hide_dist2, is_cdf=False, mean1=None, mean2=None):
    fig = go.Figure()

    # Distribution 1
    if dist_type_1 == "Discrete" and is_cdf:
        fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines+markers', name=f"{label1} 1", 
                                 line=dict(color='red', shape='hv'), marker=dict(symbol='circle', color='red')))
    elif dist_type_1 == "Discrete":
        fig.add_trace(go.Bar(x=x1, y=y1, name=f"{label1} 1", marker_color='salmon', width=0.1))
    else:
        fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name=f"{label1} 1", line=dict(color='red')))

    # Distribution 2
    if not hide_dist2:
        if dist_type_2 == "Discrete" and is_cdf:
            fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines+markers', name=f"{label2} 2", 
                                     line=dict(color='blue', shape='hv'), marker=dict(symbol='circle', color='blue')))
        elif dist_type_2 == "Discrete":
            fig.add_trace(go.Bar(x=x2, y=y2, name=f"{label2} 2", marker_color='orange'))
        else:
            fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name=f"{label2} 2", line=dict(color='orange')))

    # Add mean line for Distribution 1
    if not is_cdf and dist_type_1 == "Continuous" and mean1 is not None:
        if isinstance(x1, np.ndarray) and isinstance(y1, np.ndarray):
            closest_idx = (np.abs(x1 - mean1)).argmin()
            fig.add_shape(
                type="line",
                x0=mean1, y0=0,
                x1=mean1, y1=y1[closest_idx],
                line=dict(color="red", width=4, dash="dot"),
                name="Mean Line"
            )
            fig.add_trace(go.Scatter(x=[mean1], y=[y1[closest_idx]],
                                    mode='markers',
                                    marker=dict(color="red", size=8),
                                    showlegend=False, name=f"E[X]: {label1}"))
    # Add mean line for Distribution 2
    if not is_cdf and dist_type_2 == "Continuous" and mean2 is not None and not hide_dist2:
        if isinstance(x2, np.ndarray) and isinstance(y2, np.ndarray):
            closest_idx = (np.abs(x2 - mean2)).argmin()
            fig.add_shape(
                type="line",
                x0=mean2, y0=0,
                x1=mean2, y1=y2[closest_idx],
                line=dict(color="orange", width=4, dash="dot"),
                name="Mean Line"
            )
            fig.add_trace(go.Scatter(x=[mean2], y=[y2[closest_idx]],
                                    mode='markers',
                                    marker=dict(color="orange", size=8),
                                    showlegend=False,  name=f"E[X]: {label2}"))


    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='simple_white',
        xaxis=dict(
            showgrid=False,
            rangeslider=dict(visible=show_slider),
            type='linear',
            automargin=True
        ),
        yaxis=dict(showgrid=True),
        hovermode='closest',
    )

    st.plotly_chart(fig, use_container_width=True)

# -- App Layout --
# st.title("Probability Distribution Visualisation")
st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 10px; ">
            <h2 style="margin: 0;">Probability Distribution Visualisation</h2>
            <div style="position: relative; display: inline-block; cursor:pointer;">
                <span style="background-color:#007bff; color:white; border-radius:50%; padding:2px 7px; font-size:12px; font-weight:bold; cursor:pointer;" title="You can select one or two distributions from both continuous and discrete options. After clicking ENTER, the CDF and PMF/PDF of the chosen distributions will be displayed, with the Red line representing the first distribution and the Orange line representing the second.">
                info
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -- Main Layout --
cols = st.columns([2, 8, 2])

with cols[0]:
    st.subheader("Distribution 1")
    dist_type_1 = st.selectbox("Type of Distribution 1", ["Continuous", "Discrete"], key="type1")
    dist1 = st.selectbox("Choose Distribution to Display", continuous_dists if dist_type_1 == "Continuous" else discrete_dists, key="dist1")
    params1 = get_params(dist1, "p1")

with cols[2]:
    st.subheader("Distribution 2")
    dist_type_2 = st.selectbox("Type of Distribution 2", ["Continuous", "Discrete"], key="type2")
    dist2 = st.selectbox("Choose Distribution to Display", continuous_dists if dist_type_2 == "Continuous" else discrete_dists, key="dist2")
    params2 = get_params(dist2, "p2")
    hide_params = st.checkbox("Hide Distribution 2", value=False, key="hide_dist2")
    show_slider = st.checkbox("Show range slider", value=False, key="show_slider")


with cols[1]:
    x1, pdf1, cdf1, mean1, std1 = get_distribution_data(dist1, params1, dist_type_1)
    x2, pdf2, cdf2, mean2, std2 = get_distribution_data(dist2, params2, dist_type_2)

    label1 = f"{dist1}"
    label2 = f"{dist2}"

    if dist_type_1 == "Continuous" and dist_type_2 == "Continuous" and mean1 is not None and std1 is not None and mean2 is not None and std2 is not None:
        min_x = min(mean1 - 4 * std1, mean2 - 4 * std2)
        max_x = max(mean1 + 4 * std1, mean2 + 4 * std2)
        x_common = np.linspace(min_x, max_x, 5000)
    else:
        xmin = min(np.min(x1), np.min(x2))
        xmax = max(np.max(x1), np.max(x2))
        x_common = np.linspace(xmin, xmax, 5000)

    # st.subheader("PDF/PMF of Distributions")
    if dist_type_1 == "Discrete":
        x1_plot, pdf1_plot = x1, pdf1
    else:
        pdf1_plot = np.interp(x_common, x1, pdf1)
        x1_plot = x_common

    if dist_type_2 == "Discrete":
        x2_plot, pdf2_plot = x2, pdf2
    else:
        pdf2_plot = np.interp(x_common, x2, pdf2)
        x2_plot = x_common

    plot_distributions(x1_plot, pdf1_plot, x2_plot, pdf2_plot, label1, label2, "Value", "Density", "PDF/PMF Comparison", dist_type_1, dist_type_2, hide_params, is_cdf=False, mean1=mean1, mean2=mean2)

    # st.subheader("CDF of Distributions")
    if dist_type_1 == "Discrete":
        x1_cdf, cdf1_plot = x1, cdf1
    else:
        cdf1_plot = np.interp(x_common, x1, cdf1)
        x1_cdf = x_common
    

    if dist_type_2 == "Discrete":
        x2_cdf, cdf2_plot = x2, cdf2
    else:
        cdf2_plot = np.interp(x_common, x2, cdf2)
        x2_cdf = x_common

    plot_distributions(x1_cdf, cdf1_plot, x2_cdf, cdf2_plot, label1, label2, "Value", "Cumulative Probability", "CDF Comparison", dist_type_1, dist_type_2, hide_params, is_cdf=True)
