"""
Streamlit Web Application: Bernoulli RBM Industrial Fault Detection Showcase
=============================================================================
An interactive, polished technical showcase explaining unsupervised fault
detection across 8 industrial sensor channels using a Bernoulli Restricted
Boltzmann Machine (RBM).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration and Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RBM Industrial Fault Detection | Technical Showcase",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Custom Styling (Light background, deep navy text, restrained blue/teal accents)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #f8fafc;
        --card-bg: #ffffff;
        --navy-dark: #0f172a;
        --navy-slate: #1e293b;
        --text-muted: #64748b;
        --accent-blue: #0284c7;
        --accent-blue-light: #e0f2fe;
        --accent-teal: #0f766e;
        --accent-teal-light: #ccfbf1;
        --border-color: #e2e8f0;
        --border-subtle: #f1f5f9;
        --danger-light: #fee2e2;
        --danger-text: #991b1b;
        --success-light: #dcfce7;
        --success-text: #166534;
        --amber-light: #fef3c7;
        --amber-text: #92400e;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--navy-dark);
        background-color: var(--bg-main);
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: var(--navy-dark);
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: var(--navy-dark);
        margin-bottom: 8px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--navy-slate);
        font-weight: 500;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .problem-box {
        background: #ffffff;
        border-left: 4px solid var(--accent-blue);
        border-radius: 8px;
        padding: 14px 20px;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--navy-slate);
        margin-top: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    /* Stat Pills */
    .pill-group {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
    }
    .stat-pill {
        display: inline-flex;
        align-items: center;
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: 9999px;
        padding: 6px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--navy-slate);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .stat-pill-accent {
        background: var(--accent-blue-light);
        border-color: #bae6fd;
        color: #0369a1;
    }

    /* Cards */
    .custom-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.03);
        height: 100%;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--navy-dark);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-desc {
        font-size: 0.9rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* Pipeline Step */
    .pipeline-step {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-top: 3px solid var(--accent-blue);
        border-radius: 10px;
        padding: 16px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .step-number {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        color: var(--accent-blue);
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .step-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--navy-dark);
        margin-bottom: 6px;
    }
    .step-text {
        font-size: 0.82rem;
        color: var(--text-muted);
        line-height: 1.45;
    }

    /* Discovery Card */
    .discovery-card {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-left: 5px solid var(--accent-teal);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
    }
    .discovery-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--navy-dark);
        margin-bottom: 8px;
    }
    .discovery-body {
        font-size: 0.92rem;
        color: var(--navy-slate);
        line-height: 1.55;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-success { background: var(--success-light); color: var(--success-text); }
    .badge-danger { background: var(--danger-light); color: var(--danger-text); }
    .badge-warning { background: var(--amber-light); color: var(--amber-text); }
    .badge-info { background: var(--accent-blue-light); color: #0284c7; }

    /* Tables */
    .report-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        margin-top: 10px;
    }
    .report-table th {
        background-color: #f1f5f9;
        color: var(--navy-slate);
        font-weight: 700;
        text-align: left;
        padding: 10px 14px;
        border-bottom: 2px solid var(--border-color);
    }
    .report-table td {
        padding: 10px 14px;
        border-bottom: 1px solid var(--border-color);
        color: var(--navy-slate);
    }
    .report-table tr:hover {
        background-color: #f8fafc;
    }
    .report-table .highlight-row {
        background-color: #f0fdf4;
        font-weight: 600;
    }

    /* Code Snippet / Inline */
    code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85em;
        background-color: #f1f5f9;
        padding: 2px 6px;
        border-radius: 4px;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data & Cache Helpers
# -----------------------------------------------------------------------------
@st.cache_data
def get_method_comparison_data():
    """Returns verified experimental metrics across all historical and Phase 8 methods."""
    return [
        {
            "Method": "Original Upper 3σ Free Energy (Baseline)",
            "Precision": 0.000,
            "Recall": 0.000,
            "F1-Score": 0.000,
            "False Positives": 0,
            "Window 1 [300, 350)": "0 / 50",
            "Window 2 [420, 440)": "0 / 20",
            "Status": "Failed to flag lower-tail energy drops",
        },
        {
            "Method": "Phase 6 Lower 1st-Percentile Free Energy",
            "Precision": 0.9455,
            "Recall": 0.7429,
            "F1-Score": 0.8320,
            "False Positives": 3,
            "Window 1 [300, 350)": "50 / 50",
            "Window 2 [420, 440)": "2 / 20",
            "Status": "Detects saturation, misses noise window",
        },
        {
            "Method": "Phase 7 Bilateral 3σ Free Energy",
            "Precision": 1.0000,
            "Recall": 0.6143,
            "F1-Score": 0.7611,
            "False Positives": 0,
            "Window 1 [300, 350)": "43 / 50",
            "Window 2 [420, 440)": "0 / 20",
            "Status": "Conservative two-sided energy bound",
        },
        {
            "Method": "Phase 7 Reconstruction Error 3σ (Sample MSE)",
            "Precision": 0.9365,
            "Recall": 0.8429,
            "F1-Score": 0.8872,
            "False Positives": 4,
            "Window 1 [300, 350)": "50 / 50",
            "Window 2 [420, 440)": "9 / 20",
            "Status": "MSE dilution on localized noise spikes",
        },
        {
            "Method": "Phase 8 Per-Channel 3σ (Union Rule)",
            "Precision": 0.6667,
            "Recall": 1.0000,
            "F1-Score": 0.8000,
            "False Positives": 35,
            "Window 1 [300, 350)": "50 / 50",
            "Window 2 [420, 440)": "20 / 20",
            "Status": "100% recall, high false-positive penalty",
        },
        {
            "Method": "Phase 8 Per-Channel 99th %ile (Union Rule)",
            "Precision": 0.8333,
            "Recall": 1.0000,
            "F1-Score": 0.9091,
            "False Positives": 14,
            "Window 1 [300, 350)": "50 / 50",
            "Window 2 [420, 440)": "20 / 20",
            "Status": "100% recall, 14 multi-test false alarms",
        },
        {
            "Method": "Phase 8 Maximum Normalized Residual (Max-Z)",
            "Precision": 1.0000,
            "Recall": 1.0000,
            "F1-Score": 1.0000,
            "False Positives": 0,
            "Window 1 [300, 350)": "50 / 50",
            "Window 2 [420, 440)": "20 / 20",
            "Status": "Optimal sample-level extreme-value bound",
        },
    ]


@st.cache_data
def get_sensor_diagnostics():
    """Returns verified Phase 8 per-sensor residual diagnostics."""
    return [
        {"Index": 0, "Sensor": "Bearing_Vibration_X", "Train Mean": 0.0249, "Train Q99": 0.1673, "Normal Test": 0.0259, "W1 Mean [300, 350)": 0.2264, "W2 Mean [420, 440)": 0.0224, "W1 Flagged": "50 / 50", "W2 Flagged": "0 / 20"},
        {"Index": 1, "Sensor": "Bearing_Vibration_Y", "Train Mean": 0.0204, "Train Q99": 0.1426, "Normal Test": 0.0177, "W1 Mean [300, 350)": 0.0218, "W2 Mean [420, 440)": 0.1095, "W1 Flagged": "1 / 50", "W2 Flagged": "10 / 20"},
        {"Index": 2, "Sensor": "Motor_Winding_Temp", "Train Mean": 0.0216, "Train Q99": 0.1468, "Normal Test": 0.0202, "W1 Mean [300, 350)": 0.2245, "W2 Mean [420, 440)": 0.0196, "W1 Flagged": "50 / 50", "W2 Flagged": "0 / 20"},
        {"Index": 3, "Sensor": "Coolant_Pressure", "Train Mean": 0.0220, "Train Q99": 0.1526, "Normal Test": 0.0193, "W1 Mean [300, 350)": 0.0223, "W2 Mean [420, 440)": 0.1627, "W1 Flagged": "1 / 50", "W2 Flagged": "18 / 20"},
        {"Index": 4, "Sensor": "Pump_Flow_Rate", "Train Mean": 0.0198, "Train Q99": 0.1389, "Normal Test": 0.0171, "W1 Mean [300, 350)": 0.2048, "W2 Mean [420, 440)": 0.0106, "W1 Flagged": "50 / 50", "W2 Flagged": "0 / 20"},
        {"Index": 5, "Sensor": "Motor_Current_PhaseA", "Train Mean": 0.0227, "Train Q99": 0.1514, "Normal Test": 0.0234, "W1 Mean [300, 350)": 0.0243, "W2 Mean [420, 440)": 0.2541, "W1 Flagged": "3 / 50", "W2 Flagged": "20 / 20"},
        {"Index": 6, "Sensor": "Exhaust_Gas_Temp", "Train Mean": 0.0227, "Train Q99": 0.1585, "Normal Test": 0.0210, "W1 Mean [300, 350)": 0.1138, "W2 Mean [420, 440)": 0.0333, "W1 Flagged": "19 / 50", "W2 Flagged": "1 / 20"},
        {"Index": 7, "Sensor": "Lube_Oil_Viscosity", "Train Mean": 0.0213, "Train Q99": 0.1473, "Normal Test": 0.0208, "W1 Mean [300, 350)": 0.0277, "W2 Mean [420, 440)": 0.1268, "W1 Flagged": "3 / 50", "W2 Flagged": "11 / 20"},
    ]


# =============================================================================
# SECTION A: HERO — THE PROJECT IN ONE GLANCE
# =============================================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Industrial Fault Detection with an RBM</div>
        <div class="hero-subtitle">
            An unsupervised energy-based model that learns normal multi-channel sensor dynamics
            and autonomously detects complex industrial failures without ever seeing fault examples during training.
        </div>
        <div class="pill-group">
            <span class="stat-pill stat-pill-accent">⚙️ 8 Industrial Sensor Channels</span>
            <span class="stat-pill">📈 2,000 Normal Training Samples</span>
            <span class="stat-pill">🧪 500 Test Samples</span>
            <span class="stat-pill">🚨 70 Injected Faults (2 Regimes)</span>
            <span class="stat-pill">🧠 Bernoulli RBM (5 Hidden Units)</span>
            <span class="stat-pill">🔒 Zero Test Labels in Training</span>
        </div>
        <div class="problem-box">
            "Can a machine learning model learn what normal machine behavior looks like—and identify faults without seeing fault examples during training?"
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION B: HOW IT WORKS
# =============================================================================
st.subheader("1. How It Works: The Detection Pipeline")
st.markdown(
    "The system operates in a strictly unsupervised paradigm. An operational machine's baseline telemetry "
    "is modeled as a joint distribution, allowing subsequent anomalies to be flagged via statistical divergence."
)

p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)

with p_col1:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">Step 01</div>
            <div class="step-title">Sensors</div>
            <div class="step-text">Simulates 8 correlated telemetry channels driven by latent physical machinery factors and Gaussian noise.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p_col2:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">Step 02</div>
            <div class="step-title">Normalize</div>
            <div class="step-text">Min-max scales sensor readings to [0, 1] using training boundaries only, avoiding test data leakage.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p_col3:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">Step 03</div>
            <div class="step-title">RBM Train</div>
            <div class="step-text">A 2-layer neural model learns multi-sensor correlations via Contrastive Divergence (CD-1) on normal telemetry.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p_col4:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">Step 04</div>
            <div class="step-title">Score</div>
            <div class="step-text">Measures discrepancy using analytical free energy, visible reconstruction error MSE, or per-channel residuals.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p_col5:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">Step 05</div>
            <div class="step-title">Threshold</div>
            <div class="step-text">Calibrates statistical decision limits (e.g. 3σ, 99th percentile) strictly on training telemetry scores.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p_col6:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">Step 06</div>
            <div class="step-title">Evaluate</div>
            <div class="step-text">Post-hoc benchmarking using ground-truth test labels to compute precision, recall, and false alarm rates.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("🔍 Essential Technical Details & Hyperparameters"):
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        st.markdown(
            r"""
            - **Model Architecture**: Bernoulli Restricted Boltzmann Machine ($V=8$ visible units, $H=5$ hidden units).
            - **Training Algorithm**: Contrastive Divergence with 1 Gibbs sampling step (CD-1).
            - **Hyperparameters**: Learning rate $\eta = 0.01$, batch size $= 32$, epochs $= 50$.
            - **Random Seed**: Fixed at `42` for complete deterministic reproducibility across CPU runs.
            """
        )
    with t_c2:
        st.markdown(
            r"""
            - **Preprocessing**: Feature-wise min-max scaling fitted strictly on $X_{\text{train}}$: $[0, 1]$.
            - **Reconstruction Mode**: Deterministic probability reconstruction $\hat{v} = \sigma(h W + b)$ to avoid stochastic noise during evaluation.
            - **Threshold Principle**: Derived strictly from normal training residuals ($R_{\text{train}}$) without test label exposure.
            """
        )


# =============================================================================
# SECTION C: UNDERSTANDING THE DATA
# =============================================================================
st.markdown("---")
st.subheader("2. Understanding the Telemetry & Injected Faults")
st.markdown(
    "The system monitors eight continuous sensor channels representing rotating machinery subsystems. "
    "To test detection capabilities, two physically distinct fault mechanisms were injected into the test sequence."
)

d_col1, d_col2 = st.columns([1, 1])

with d_col1:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">📡 Monitored Sensor Channels</div>
            <div class="card-desc">
                Derived from a linear mixture of 3 latent physical variables (torque, speed, ambient heat):
                <ul style="margin-top: 8px; margin-bottom: 0; padding-left: 20px;">
                    <li><code>Bearing_Vibration_X</code> (Ch 0) & <code>Bearing_Vibration_Y</code> (Ch 1)</li>
                    <li><code>Motor_Winding_Temp</code> (Ch 2) & <code>Coolant_Pressure</code> (Ch 3)</li>
                    <li><code>Pump_Flow_Rate</code> (Ch 4) & <code>Motor_Current_PhaseA</code> (Ch 5)</li>
                    <li><code>Exhaust_Gas_Temp</code> (Ch 6) & <code>Lube_Oil_Viscosity</code> (Ch 7)</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with d_col2:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">🚨 Injected Fault Regimes (Test Set Only)</div>
            <div class="card-desc">
                <p><b>Fault Window 1: Indices [300, 350) — 50 Samples</b><br>
                <em>Sensor Drift / Saturation:</em> Channels 0, 2, 4, 6 saturate strongly near 1.0, simulating systematic subsystem overheating and bearing lockup.</p>
                <p style="margin-bottom: 0;"><b>Fault Window 2: Indices [420, 440) — 20 Samples</b><br>
                <em>Localized Noise Spikes:</em> Channels 1, 3, 5, 7 receive high-variance uncorrelated Gaussian noise, simulating electrical instability and sensor chatter.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION D: IMPLEMENTATION — WHAT WAS BUILT
# =============================================================================
st.markdown("---")
st.subheader("3. Modular Architecture Overview")
st.markdown(
    "The project was engineered modularly with strict test-driven development (62 unit tests passing). "
    "Here is how the system components interact:"
)

arch_c1, arch_c2, arch_c3 = st.columns(3)

with arch_c1:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">📦 Data & Baseline</div>
            <div class="card-desc">
                <b><code>data_generator.py</code></b>: Generates synthetic multi-channel physics and fits training scalers.<br><br>
                <b><code>rbm.py</code></b>: CPU-based Bernoulli RBM with CD-1 parameter updates, free energy, and reconstruction.<br><br>
                <b><code>anomaly_detector.py</code></b>: Original Phase 5 baseline using analytical free-energy upper 3σ thresholding.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with arch_c2:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">🔬 Research & Extensions</div>
            <div class="card-desc">
                <b><code>detection_experiments.py</code></b>: Phase 6 diagnostic proving anomalies cause downward energy shifts.<br><br>
                <b><code>phase7_experiments.py</code></b>: Evaluates bilateral free energy and sample-level reconstruction error MSE.<br><br>
                <b><code>phase8_experiments.py</code></b>: Computes channel-wise squared residuals and maximum normalized scoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with arch_c3:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">📊 Reporting & Verification</div>
            <div class="card-desc">
                <b><code>visualization.py</code></b>: Headless Matplotlib report generators producing publication-ready PNGs.<br><br>
                <b><code>test_*.py</code></b> (7 suites): 62 unit tests enforcing label independence, safe variance, and deterministic outputs.<br><br>
                <b><code>run_phase*.py</code></b>: Standalone benchmark runners for reproducible execution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Key distinction explanation
st.markdown(
    r"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 24px; margin-top: 12px;">
        <h4 style="margin-top: 0; color: #0f172a;">Key Distinction: Three Generations of Scoring Mechanisms</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 10px;">
            <div>
                <b style="color: #0369a1;">1. Analytical Free Energy $F(v)$</b><br>
                <span style="font-size: 0.88rem; color: #475569;">
                    Measures the negative log-unnormalized probability of visible vector $v$. Produces a single aggregate scalar per sample.
                </span>
            </div>
            <div>
                <b style="color: #0f766e;">2. Sample Reconstruction Error MSE</b><br>
                <span style="font-size: 0.88rem; color: #475569;">
                    Measures average Euclidean distance across all 8 channels $\frac{1}{V}\|v - \hat{v}\|^2$. Can dilute single-channel faults.
                </span>
            </div>
            <div>
                <b style="color: #0d9488;">3. Channel-Wise Residuals $R_{i,j}$</b><br>
                <span style="font-size: 0.88rem; color: #475569;">
                    Inspects squared reconstruction errors $(X_{i,j} - \hat{X}_{i,j})^2$ independently per sensor channel, isolating localized disruptions.
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION E: RESULTS — SHOW THE EVIDENCE
# =============================================================================
st.markdown("---")
st.subheader("4. Experimental Results: Historical Methods vs. Phase 8")
st.markdown(
    "All thresholds were derived strictly from normal training telemetry without using test labels. "
    "Below are the verified empirical results across every research phase."
)

results_data = get_method_comparison_data()

# Render styled comparison table
table_rows = []
for row in results_data:
    is_top = "Maximum Normalized" in row["Method"]
    row_class = ' class="highlight-row"' if is_top else ""
    prec_str = f"{row['Precision']*100:.1f}%"
    rec_str = f"{row['Recall']*100:.1f}%"
    f1_str = f"{row['F1-Score']:.3f}"
    table_rows.append(
        f'<tr{row_class}>'
        f'<td><b>{row["Method"]}</b></td>'
        f'<td style="text-align: right;">{prec_str}</td>'
        f'<td style="text-align: right;">{rec_str}</td>'
        f'<td style="text-align: right;">{f1_str}</td>'
        f'<td style="text-align: right;">{row["False Positives"]}</td>'
        f'<td style="text-align: center;">{row["Window 1 [300, 350)"]}</td>'
        f'<td style="text-align: center;">{row["Window 2 [420, 440)"]}</td>'
        f'</tr>'
    )

table_html = (
    '<table class="report-table">'
    '<thead><tr>'
    '<th>Method / Scoring Rule</th>'
    '<th style="text-align: right;">Precision</th>'
    '<th style="text-align: right;">Recall</th>'
    '<th style="text-align: right;">F1-Score</th>'
    '<th style="text-align: right;">False Positives</th>'
    '<th style="text-align: center;">Window 1 (/50)</th>'
    '<th style="text-align: center;">Window 2 (/20)</th>'
    '</tr></thead>'
    '<tbody>'
    + "".join(table_rows) +
    '</tbody></table>'
)
st.markdown(table_html, unsafe_allow_html=True)
st.caption("⚠️ Note: Results derived on a controlled synthetic industrial sensor test set with fixed seed 42, not a field deployment.")

# Visual Performance Chart
chart_col1, chart_col2 = st.columns([1.2, 0.8])

with chart_col1:
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    methods = [
        "Base Upper 3σ",
        "P6 Lower 1%",
        "P7 Bilateral 3σ",
        "P7 Recon 3σ",
        "P8 Chan 3σ Un.",
        "P8 Chan 99% Un.",
        "P8 Max-Z (Top-1)",
    ]
    f1_scores = [0.000, 0.832, 0.761, 0.887, 0.800, 0.909, 1.000]
    recalls = [0.000, 0.743, 0.614, 0.843, 1.000, 1.000, 1.000]

    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width / 2, f1_scores, width, label="F1-Score", color="#0284c7")
    ax.bar(x + width / 2, recalls, width, label="Recall", color="#0f766e", alpha=0.85)

    ax.set_ylabel("Metric Value", fontsize=9, color="#0f172a", fontweight="600")
    ax.set_title("Method Performance: F1-Score vs. Recall Across Generations", fontsize=10, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8), dpi=150)
    fp_counts = [0, 3, 0, 4, 35, 14, 0]
    colors = ["#94a3b8", "#f59e0b", "#94a3b8", "#f59e0b", "#ef4444", "#f97316", "#10b981"]

    bars = ax2.bar(methods, fp_counts, color=colors, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, int(yval), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_ylabel("False Positive Count (430 Normal)", fontsize=9, color="#0f172a", fontweight="600")
    ax2.set_title("The False Positive Penalty: Union vs. Max-Z", fontsize=10, fontweight="bold", pad=12)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax2.set_ylim(0, 42)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    fig2.patch.set_facecolor("#f8fafc")
    ax2.set_facecolor("#ffffff")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# =============================================================================
# SECTION F: THE MOST IMPORTANT DISCOVERIES
# =============================================================================
st.markdown("---")
st.subheader("5. Key Scientific Discoveries")
st.markdown("Our progressive investigations uncovered four core insights into unsupervised energy-based fault detection:")

disc_col1, disc_col2 = st.columns(2)

with disc_col1:
    st.markdown(
        r"""
        <div class="discovery-card">
            <div class="discovery-title">🎯 Discovery 1: The Scoring Direction Matters</div>
            <div class="discovery-body">
                The baseline upper-tail free-energy threshold ($\mu + 3\sigma$) detected <b>0 of 70 faults</b>.
                In a Bernoulli RBM, when sensor values saturate near 1.0 (Fault Window 1), the linear visible bias
                $-b^T v$ and softplus activations become large. Because these terms are subtracted,
                <b>free energy shifts drastically downward into the lower tail</b>.
                A threshold is only useful when its mathematical direction matches physical score movement.
            </div>
        </div>
        <div class="discovery-card" style="border-left-color: #0284c7;">
            <div class="discovery-title">⚖️ Discovery 3: More Detections ≠ Better Detector</div>
            <div class="discovery-body">
                Testing each channel individually under a naive union rule (flagging if <em>any</em> channel exceeds threshold)
                reached 100% recall on both windows, but introduced <b>35 false alarms in Method A and 14 in Method B</b>.
                In a real-world plant, alarm fatigue causes operators to ignore alerts.
                By switching to <b>Maximum Normalized Residual scoring (Method C)</b>, the sample-level false alarm rate is bounded
                to &le; 1%, achieving 100% precision and 100% recall.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with disc_col2:
    st.markdown(
        r"""
        <div class="discovery-card" style="border-left-color: #0d9488;">
            <div class="discovery-title">🔍 Discovery 2: Aggregate Scores Hide Localized Faults</div>
            <div class="discovery-body">
                Scalar reconstruction error MSE detected 50/50 saturation faults in Window 1, but only <b>9 of 20</b> noise spikes in Window 2.
                Because the noise spikes were confined to odd-indexed sensors (especially <code>Motor_Current_PhaseA</code> and <code>Coolant_Pressure</code>),
                the four unaffected sensors diluted the overall average.
                Channel-wise residual scoring exposed individual channel residuals jumping by <b>8.4× to 10.9×</b> over normal operating levels.
            </div>
        </div>
        <div class="discovery-card" style="border-left-color: #6366f1;">
            <div class="discovery-title">📐 Discovery 4: Model Assumptions Matter</div>
            <div class="discovery-body">
                A Bernoulli RBM assumes visible units are binary probabilities. Min-max scaling continuous sensor
                signals to $[0, 1]$ is a standard engineering approximation, but probability reconstructions
                $\sigma(h W + b)$ are naturally bounded in $(0, 1)$.
                For unconstrained continuous industrial telemetry with drifting operating points,
                a <b>Gaussian-Bernoulli RBM</b> provides the mathematically rigorous foundation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION G: CONCLUSIONS & LIMITATIONS
# =============================================================================
st.markdown("---")
st.subheader("6. Conclusions & Scientific Safeguards")

st.markdown(
    """
    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #166534; font-size: 1.15rem;">Core Takeaway</h4>
        <p style="font-size: 1.02rem; color: #14532d; font-weight: 500; margin-bottom: 0;">
            "The experiments show that anomaly detection depends not just on the model, but also on the scoring method, threshold calibration, and sensor-level information."
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

t_c1, t_c2, t_c3 = st.columns(3)
with t_c1:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">1. Train-Only Thresholds</div>
            <div class="card-desc">
                Thresholds must be established strictly from normal training residuals. Using test labels to select thresholds violates unsupervised integrity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t_c2:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">2. Channel Localization</div>
            <div class="card-desc">
                Per-sensor residual decomposition isolates physical subsystems and prevents localized anomalies from being diluted by healthy channels.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t_c3:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">3. Extreme-Value Calibration</div>
            <div class="card-desc">
                Taking sample-level maximum normalized residuals bounds multi-testing false alarms without sacrificing sensitivity to concentrated disturbances.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("⚠️ Industrial Limitations & Deployment Disclosures (Must Read)"):
    st.markdown(
        """
        - **Synthetic Telemetry**: The data and injected disturbances are synthetic. Real industrial plants exhibit sensor degradation, operating regime switches, and non-stationary ambient conditions.
        - **Controlled Dataset Scope**: The 100% precision and recall was achieved on this specific seeded test set (`seed=42`). It demonstrates proof-of-concept capability, not guaranteed field performance.
        - **Single-Seed Evaluation**: The findings represent controlled comparisons on one fixed dataset; multi-seed robustness studies are necessary prior to operational deployment.
        - **Multiple-Testing Scaling**: In plants with hundreds or thousands of channels, extreme-value distributions require formal Bonferroni or False Discovery Rate (FDR) control to avoid false alarm inflation.
        - **Not Production-Ready**: This prototype is a scientific benchmark demonstrating algorithmic trade-offs, not a validated turn-key industrial safety system.
        """
    )


# =============================================================================
# SECTION H: VISUAL EVIDENCE
# =============================================================================
st.markdown("---")
st.subheader("7. Visual Evidence & Diagnostic Artifacts")
st.markdown("Inspect the generated diagnostic report figures across the research progression:")

tab1, tab2, tab3 = st.tabs(["Phase 8 (Channel Residuals)", "Phase 7 (Reconstruction & Bilateral)", "Phase 5 Baseline"])

def display_report_image(img_path: Path, caption: str):
    if img_path.exists():
        st.image(str(img_path), caption=caption, use_column_width=True)
    else:
        st.warning(f"Report image not found at `{img_path}`. Run the corresponding pipeline script to generate it.")

with tab1:
    st.markdown(
        "**Phase 8 Diagnostic Report**: Shows raw sensor telemetry, channel-wise squared residual heatmap "
        "revealing localized subsystem disturbances, per-channel threshold bars, and overall method comparison."
    )
    display_report_image(
        BASE_DIR / "phase8_anomaly_report.png",
        "Figure 1: Phase 8 Channel-Wise Residual Anomaly Report (Localized Disturbance Heatmap & Method Comparison)",
    )

with tab2:
    st.markdown(
        "**Phase 7 Diagnostic Report**: Demonstrates why bilateral free-energy thresholding and sample MSE "
        "reconstruction error outperformed unilateral upper-tail scoring."
    )
    display_report_image(
        BASE_DIR / "phase7_anomaly_report.png",
        "Figure 2: Phase 7 Advanced Anomaly Report (Bilateral Energy, Scalar MSE, and Combined Logic)",
    )

with tab3:
    st.markdown(
        "**Phase 5 Baseline Report**: Shows original upper-tail free-energy scoring missing lower-tail saturation shifts."
    )
    display_report_image(
        BASE_DIR / "anomaly_report.png",
        "Figure 3: Phase 5 Original Anomaly Report (Upper 3-Sigma Threshold Failure)",
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 20px 0;">
        <b>RBM Industrial Fault Detection Case Study</b> &bull; CPU-Only Deterministic Prototype &bull;
        Built with NumPy, Matplotlib & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
