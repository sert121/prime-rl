import streamlit as st

st.set_page_config(page_title="Qwen Memory Calculator", layout="centered")

st.title("🧮 Qwen RL Memory Calculator")
st.write("""
This tool estimates GPU memory consumption for RL training with Qwen models.

It computes:
- **Activation memory**
- **Logits memory**
- **Total microbatch accumulation memory**
- **Total GPU memory required**

Formulas are shown next to each section.
""")

# -----------------------------
# Inputs
# -----------------------------
st.header("Inputs")

hidden_size = st.number_input("Hidden size", value=1536)
vocab_size = st.number_input("Vocab size", value=151936)
seq_len = st.number_input("Sequence Length", value=2048)
batch_size = st.number_input("Global Batch Size", value=128)
micro_batch_size = st.number_input("Micro Batch Size", value=4)
dtype = st.selectbox("Activation dtype", ["float32", "bfloat16"])

bytes_per_elem = 4 if dtype == "float32" else 2

# -----------------------------
# Calculations
# -----------------------------
num_micro_batches = batch_size // micro_batch_size

# Activation memory (approx)
activation_mem = (
    micro_batch_size * seq_len * hidden_size * bytes_per_elem * 4  # factor 4 for attention + MLP activations
)
activation_mem_gb = activation_mem / (1024**3)

# Logits memory
logits_mem = (
    micro_batch_size * seq_len * vocab_size * bytes_per_elem
)
logits_mem_gb = logits_mem / (1024**3)

# Microbatch accumulation memory
total_activation_mem = activation_mem_gb * num_micro_batches
total_logits_mem = logits_mem_gb * num_micro_batches

total_mem_gb = total_activation_mem + total_logits_mem

# -----------------------------
# Display results
# -----------------------------
st.header("Results")

st.subheader("1. Activation Memory (per microbatch)")
st.latex(
    r"\text{activation\_bytes} = B_{\text{micro}} \times L \times H \times \text{bytes\_per\_elem} \times 4"
)
st.write(f"**≈ {activation_mem_gb:.2f} GB** per microbatch")

st.subheader("2. Logits Memory (per microbatch)")
st.latex(
    r"\text{logits\_bytes} = B_{\text{micro}} \times L \times V \times \text{bytes\_per\_elem}"
)
st.write(f"**≈ {logits_mem_gb:.2f} GB** per microbatch")

st.subheader("3. With Gradient Accumulation")
st.latex(
    r"\text{num\_microbatches} = \frac{B_{\text{global}}}{B_{\text{micro}}}"
)
st.write(f"Microbatches = **{num_micro_batches}**")

st.write("### Total Activation Memory")
st.write(f"≈ **{total_activation_mem:.2f} GB**")

st.write("### Total Logits Memory")
st.write(f"≈ **{total_logits_mem:.2f} GB**")

st.subheader("4. Estimated Total Memory")
st.latex(r"\text{total} = (\text{activation} + \text{logits}) \times \text{microbatches}")
st.write(f"# **Total Estimated: {total_mem_gb:.2f} GB**")

# Safety note
st.info("""
### ⚠ Notes
- These are **upper-bound estimates**.
- Real models have optimizer states, KV caches, gradients, and fragmentation overhead.
- Expect actual memory use to be **1.2×–1.8×** this estimate.
""")
