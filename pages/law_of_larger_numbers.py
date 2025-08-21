import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("Distribution Experiment Simulator")

# --- Sidebar for input ---
st.sidebar.header("Configuration")

preset = st.sidebar.selectbox("Select Distribution", [
    "Fair Coin (Head:0.5, Tail:0.5)",
    "Fair 6-sided Die (1/6 each)",
    "Red:6, Blue:4 (Frequencies)",
    "Custom Distribution (Frequencies)"
])
 
def validate_probabilities(values):
    total = sum(values)
    if total == 0:
        st.error("Sum of values must be greater than 0.")
        st.stop()
    elif total >= 1 + 1e-6:
        st.error(f"❌ The total sum of probabilities is {total:.3f}, which exceeds 1. Please adjust your inputs.")
        st.stop()
    return [v / total for v in values] if abs(total - 1.0) > 1e-6 else values

def validate_frequencies(values):
    total = sum(values)
    if total == 0:
        st.error("Sum of frequencies must be greater than 0.")
        st.stop()
    return [v / total for v in values]

def is_probability_mode(values):
    return all(v <= 1.0 for v in values)

def parse_one_line_input(text):
    try:
        items = [item.strip() for item in text.split(',')]
        names = []
        values = []
        for idx, item in enumerate(items):
            if ':' in item:
                name, val = item.split(':')
            else:
                name, val = f"Outcome {idx+1}", item
            names.append(name.strip())
            values.append(float(eval(val.strip())))
        return names, values
    except:
        st.sidebar.error("Invalid format. Use format like A:1, B:2")
        st.stop()

def parse_step_by_step(names_text, preset):
    names = [name.strip() for name in names_text.split(',')]
    values = []
    default_map = {
        "Fair Coin (Head:0.5, Tail:0.5)": {"Head": 0.5, "Tail": 0.5},
        "Fair 6-sided Die (1/6 each)": {f"Die {i+1}": 1/6 for i in range(6)},
        "Red:6, Blue:4 (Frequencies)": {"Red": 6, "Blue": 4},
        "Custom Distribution (Frequencies)": {f"Outcome {i+1}": 1.0 for i in range(len(names))}
    }
    defaults = default_map.get(preset, {})
    for name in names:
        val = st.sidebar.number_input(f"Value for {name}", value=float(defaults.get(name, 1.0)), step=1.0 if not is_probability_mode(list(defaults.values())) else 0.1, key=name)
        values.append(val)
    return names, values

def get_default_names(preset):
    if preset == "Fair Coin (Head:0.5, Tail:0.5)":
        return "Head, Tail"
    elif preset == "Fair 6-sided Die (1/6 each)":
        return ", ".join([f"Die {i+1}" for i in range(6)])
    elif preset == "Red:6, Blue:4 (Frequencies)":
        return "Red, Blue"
    elif preset == "Custom Distribution (Frequencies)":
        return "Outcome 1, Outcome 2, Outcome 3, Outcome 4"
    return "A, B"

def get_default_input(preset):
    return {
        "Fair Coin (Head:0.5, Tail:0.5)": "Head:0.5, Tail:0.5",
        "Fair 6-sided Die (1/6 each)": ", ".join([f"Die {i+1}:1/6" for i in range(6)]),
        "Red:6, Blue:4 (Frequencies)": "Red:6, Blue:4",
        "Custom Distribution (Frequencies)": "Outcome 1:1, Outcome 2:1, Outcome 3:1, Outcome 4:3"
    }[preset]

input_format = st.sidebar.radio("Input Format", ["One-line", "Step-by-step"])

if input_format == "One-line":
    one_line = st.sidebar.text_input("One-line Input:", get_default_input(preset))
    outcome_names, values = parse_one_line_input(one_line)
else:
    default_names = get_default_names(preset)
    names_text = st.sidebar.text_input("Outcome names (comma-separated)", default_names)
    if st.sidebar.button("Generate Rows"):
        pass
    outcome_names, values = parse_step_by_step(names_text, preset)

if is_probability_mode(values):
    probs = validate_probabilities(values)
else:
    probs = validate_frequencies(values)

trials = st.sidebar.number_input("Trials:", min_value=1, value=10)
rewind = st.sidebar.slider("Experiment Rewind:", 1, int(trials), value=int(trials))

# ✅ Fix: only simulate the number of trials needed (rewind)
freq = {name: 0 for name in outcome_names}
for _ in range(rewind):
    outcome = random.choices(outcome_names, weights=probs)[0]
    freq[outcome] += 1
exp_probs = [freq[name] / rewind for name in outcome_names]

# --- Sorting for Consistent Output ---
sorted_labels = sorted(outcome_names)
sorted_probs = [probs[outcome_names.index(lbl)] for lbl in sorted_labels]
sorted_exp_probs = [exp_probs[outcome_names.index(lbl)] for lbl in sorted_labels]

# --- Pie Chart (Ground Truth Only) ---
def draw_pie_chart(labels, probs):
    pastel_colors = ['#ffc0cb', '#add8e6', '#ffe4b5', '#b0e0e6', '#fcd5ce', '#d0f4de']
    fig, ax = plt.subplots()
    ax.pie(probs, labels=labels, colors=pastel_colors[:len(labels)], startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    plt.title("Ground Truth Distribution")
    return fig

import plotly.graph_objects as go

# --- Comparison Piechart (Donut Style) ---
def draw_comparison_pie_plotly(labels, ground_truth, experiment_result):
    fig = go.Figure()

    # Outer Ring
    fig.add_trace(go.Pie(
        labels=labels,
        values=ground_truth,
        name="Ground Truth",
        hole=0.5,
        direction='clockwise',
        sort=False,
        marker=dict(colors=['#ffcccc', '#cce6ff']),
        textinfo='none',
        hoverinfo='label+percent+name',
        showlegend=False,
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    # Inner Ring
    fig.add_trace(go.Pie(
        labels=labels,
        values=experiment_result,
        name="Experiment Result",
        hole=0.75,
        direction='clockwise',
        sort=False,
        marker=dict(colors=['#ff9999', '#99ccff']),
        textinfo='none',
        hoverinfo='label+percent+name',
        showlegend=True,
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        title=dict(
            text=f"Experiment result after trial {rewind}",
            font=dict(size=18, color="#222"),
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            orientation='h',
            x=0.3,
            y=1.1,
            font=dict(color="#222")
        ),
        font=dict(color="#222"),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=100),
        height=500,
        width=500
    )
    return fig


# --- Bar Chart Comparison ---
def draw_comparison_bar_plotly(labels, ground_truth, experiment_result):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=ground_truth,
        name='Ground Truth',
        marker_color='#ffc0cb',
        text=[f"{v*100:.1f}%" for v in ground_truth],
        textposition='outside',
        hovertemplate='Ground Truth: %{text}<extra></extra>',
    ))

    fig.add_trace(go.Bar(
        x=labels,
        y=experiment_result,
        name='Experiment Result',
        marker_color='#ff7597',
        text=[f"{v*100:.1f}%" for v in experiment_result],
        textposition='outside',
        hovertemplate='Experiment Result: %{text}<extra></extra>',
    ))

    fig.update_layout(
    title=dict(
        text='Bar Comparison: Ground Truth vs Experiment',
        font=dict(size=18, color="#222"),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title=dict(text='Outcome', font=dict(color="#222")),
        tickfont=dict(color="#222")
    ),
    yaxis=dict(
        title=dict(text='Probability', font=dict(color="#222")),
        tickfont=dict(color="#222")
    ),
    font=dict(color="#222"),
    legend=dict(font=dict(color="#222")),
    paper_bgcolor='white',
    plot_bgcolor='white',
    height=500
)

    return fig


# --- Final Layout with Charts ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Piechart")
    st.plotly_chart(draw_comparison_pie_plotly(sorted_labels, sorted_probs, sorted_exp_probs), use_container_width=True)

with col2:
    st.markdown("### Bar Comparison")
    st.plotly_chart(draw_comparison_bar_plotly(sorted_labels, sorted_probs, sorted_exp_probs), use_container_width=True)

