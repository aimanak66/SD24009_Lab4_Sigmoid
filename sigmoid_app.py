import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation Function", page_icon="📈", layout="centered")
st.title("Sigmoid Activation Function")
st.write("Sigmoid maps any real-valued input to an output between 0 and 1.")
st.latex(r"\\sigma(x) = \\frac{1}{1 + e^{-x}}")

x_min = st.slider("Minimum input value", -20, 0, -10)
x_max = st.slider("Maximum input value", 1, 20, 10)
if x_min >= x_max:
    st.error("The maximum input must be greater than the minimum input.")
else:
    x = np.linspace(x_min, x_max, 400)
    y = 1 / (1 + np.exp(-x))
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Sigmoid")
    ax.axhline(0.5, linestyle="--", linewidth=0.8, label="y = 0.5")
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Input (x)")
    ax.set_ylabel("Output σ(x)")
    ax.set_title("Sigmoid Activation Function")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    value = st.number_input("Try an input value", value=0.0, step=0.5)
    output = 1 / (1 + np.exp(-value))
    st.metric("Sigmoid output", f"{output:.4f}")
    st.caption("Interpretation: the output approaches 0 for very negative inputs and 1 for very positive inputs.")
