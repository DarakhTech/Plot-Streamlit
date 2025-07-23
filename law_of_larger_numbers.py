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

# --- Bar Chart: Ground Truth ---
def draw_ground_truth_bar(labels, probs):
    pastel_colors = ['#ffc0cb', '#add8e6', '#ffe4b5', '#b0e0e6', '#fcd5ce', '#d0f4de']
    fig, ax = plt.subplots()
    ax.bar(labels, probs, color=pastel_colors[:len(labels)])
    ax.set_ylim([0, 1])
    ax.set_title("Ground Truth Distribution")
    return fig

# --- Bar Chart: Comparison ---
def draw_comparison_bar(labels, ground_truth, experiment_result):
    pastel_colors = ['#ffc0cb', '#add8e6', '#ffe4b5', '#b0e0e6', '#fcd5ce', '#d0f4de']
    solid_colors = ['#ff7597', '#69bce7', '#ffc107', '#76eec6', '#e29578', '#99d98c']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, ground_truth, width, label='Ground Truth', color=pastel_colors[:len(labels)])
    ax.bar(x + width/2, experiment_result, width, label='Experiment Result', color=solid_colors[:len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1])
    ax.set_title(f"Experiment result after trial {rewind}")
    ax.legend()
    return fig

# --- Comparison Piechart (Concentric Donuts) ---
def draw_comparison_pie(labels, ground_truth, experiment_result):
    colors_outer = ['#ffc0cb', '#add8e6', '#ffe4b5', '#b0e0e6', '#fcd5ce', '#d0f4de']
    colors_inner = ['#ff7597', '#69bce7', '#ffc107', '#76eec6', '#e29578', '#99d98c']

    fig, ax = plt.subplots()
    ax.axis('equal')

    wedges_outer, _ = ax.pie(ground_truth,
                             radius=1,
                             labels=None,
                             startangle=90,
                             colors=colors_outer[:len(labels)],
                             wedgeprops=dict(width=0.3, edgecolor='white'))

    wedges_inner, _ = ax.pie(experiment_result,
                             radius=0.7,
                             labels=None,
                             startangle=90,
                             colors=colors_inner[:len(labels)],
                             wedgeprops=dict(width=0.3, edgecolor='white'))

    plt.title(f"Experiment result after trial {rewind}", fontsize=14, weight='bold')

    plt.legend(wedges_outer,
               labels,
               title="",
               loc="upper center",
               bbox_to_anchor=(0.5, 1.15),
               ncol=len(labels),
               frameon=False)

    return fig

# --- Layout ---
col1, col2 = st.columns(2)
with col1:
    st.pyplot(draw_pie_chart(sorted_labels, sorted_probs))
with col2:
    st.pyplot(draw_comparison_pie(sorted_labels, sorted_probs, sorted_exp_probs))

col3, col4 = st.columns(2)
with col3:
    st.pyplot(draw_ground_truth_bar(sorted_labels, sorted_probs))
with col4:
    st.pyplot(draw_comparison_bar(sorted_labels, sorted_probs, sorted_exp_probs))
