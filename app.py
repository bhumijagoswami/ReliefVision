import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import json
from datetime import datetime
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ReliefVision",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #020617 !important;
    color: #f1f5f9 !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1300px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Hero banner ── */
.rv-hero {
    background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(168,85,247,0.10) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 24px;
    padding: 36px 44px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.rv-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(168,85,247,0.18), transparent 70%);
    border-radius: 50%;
}
.rv-hero h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #60a5fa, #a855f7, #ec4899);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 0 6px 0 !important;
}
.rv-hero p {
    color: rgba(255,255,255,0.65) !important;
    font-size: 1rem !important;
    margin: 0 !important;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.glass-card h3 {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    color: rgba(255,255,255,0.38) !important;
    margin: 0 0 10px 0 !important;
}

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-tile {
    flex: 1;
    min-width: 140px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 22px;
    text-align: center;
}
.metric-tile .val {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-tile .lbl {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.42);
}
.val-blue   { background: linear-gradient(90deg,#60a5fa,#818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.val-purple { background: linear-gradient(90deg,#a855f7,#ec4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.val-green  { background: linear-gradient(90deg,#22c55e,#06b6d4); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.val-red    { background: linear-gradient(90deg,#ef4444,#f97316); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.val-orange { background: linear-gradient(90deg,#f97316,#eab308); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

/* ── Severity badge ── */
.sev-badge {
    display: inline-block;
    padding: 8px 28px;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.sev-CRITICAL { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1.5px solid rgba(239,68,68,0.45); }
.sev-HIGH     { background: rgba(249,115,22,0.18); color: #fdba74; border: 1.5px solid rgba(249,115,22,0.45); }
.sev-MEDIUM   { background: rgba(234,179,8,0.18);  color: #fde047; border: 1.5px solid rgba(234,179,8,0.45); }
.sev-LOW      { background: rgba(34,197,94,0.18);  color: #86efac; border: 1.5px solid rgba(34,197,94,0.45); }

/* ── Zone table ── */
.zone-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; }
.zone-table th {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.15em; color: rgba(255,255,255,0.35);
    padding: 0 14px 6px; text-align: left;
}
.zone-table td {
    font-size: 0.9rem; color: #ffffff;
    padding: 13px 14px;
    background: rgba(255,255,255,0.04);
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.zone-table td:first-child { border-radius: 10px 0 0 10px; border-left: 1px solid rgba(255,255,255,0.05); }
.zone-table td:last-child  { border-radius: 0 10px 10px 0; border-right: 1px solid rgba(255,255,255,0.05); }

/* ── Section divider ── */
.rv-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    margin: 28px 0;
}

/* ── Section heading ── */
.rv-section-title {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.22em !important;
    color: rgba(255,255,255,0.32) !important;
    margin-bottom: 16px !important;
}

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(99,102,241,0.4) !important;
    border-radius: 16px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.7) !important;
}
.stSlider [data-baseweb="slider"] { padding: 8px 0 !important; }
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
.stButton > button {
    background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
    color: white !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 12px 36px !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.4) !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    box-shadow: 0 12px 36px rgba(59,130,246,0.6) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

DISASTER_PROFILES = {
    "Flood": {
        "channel_weights": (0.2, 0.3, 0.5),   # B,G,R — blue dominates
        "description": "Floods alter reflectance primarily in the blue channel. Sensitivity boosted for water-logged regions.",
        "icon": "🌊",
    },
    "Fire / Wildfire": {
        "channel_weights": (0.1, 0.2, 0.7),   # red dominates
        "description": "Fire scars and smoke create strong red-channel shifts. SSIM weighted toward burn signatures.",
        "icon": "🔥",
    },
    "Earthquake": {
        "channel_weights": (0.33, 0.34, 0.33), # uniform — structural collapse
        "description": "Seismic damage appears as uniform pixel loss across all channels. Equal-weight SSIM used.",
        "icon": "🏚️",
    },
    "Cyclone / Storm": {
        "channel_weights": (0.25, 0.45, 0.30), # green dominates (vegetation loss)
        "description": "Storm damage strips vegetation. Green-channel weight elevated to capture canopy change.",
        "icon": "🌀",
    },
}

SEVERITY_LEVELS = [
    {"label": "CRITICAL", "threshold": 50, "color": (220, 38, 38),   "badge": "sev-CRITICAL", "action": "🚨 Immediate rescue & medical deployment required — P1 response."},
    {"label": "HIGH",     "threshold": 30, "color": (249, 115, 22),  "badge": "sev-HIGH",     "action": "⚠️ Deploy relief teams and begin structural assessment — P2 response."},
    {"label": "MEDIUM",   "threshold": 15, "color": (234, 179, 8),   "badge": "sev-MEDIUM",   "action": "📡 Ground survey and logistics staging required — P3 response."},
    {"label": "LOW",      "threshold": 0,  "color": (34, 197, 94),   "badge": "sev-LOW",      "action": "🟢 Low impact zone. Remote monitoring sufficient — P4 response."},
]

def classify_zone(local_pct):
    for lvl in SEVERITY_LEVELS:
        if local_pct >= lvl["threshold"]:
            return lvl
    return SEVERITY_LEVELS[-1]

def weighted_ssim(gray1, gray2, img1_rgb, img2_rgb, weights):
    """Compute weighted per-channel SSIM."""
    scores = []
    diffs  = []
    for ch, w in zip(range(3), weights):
        s, d = ssim(img1_rgb[:,:,ch], img2_rgb[:,:,ch], full=True)
        scores.append(s * w)
        diffs.append(d * w)
    combined_score = sum(scores) / sum(weights)
    combined_diff  = np.sum(diffs, axis=0) / sum(weights)
    # Also blend with grayscale SSIM for stability
    score_gray, diff_gray = ssim(gray1, gray2, full=True)
    final_score = 0.6 * combined_score + 0.4 * score_gray
    final_diff  = 0.6 * combined_diff  + 0.4 * diff_gray
    return final_score, final_diff

def build_heatmap(diff_norm, thresh_mask):
    """Colour heatmap: blue(safe) → yellow → red(critical)."""
    heatmap = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Zero out undamaged areas
    heatmap[thresh_mask == 0] = [10, 20, 40]
    return heatmap

def confidence_score(ssim_val, damage_pct, n_zones):
    """Heuristic confidence: 0-100."""
    ssim_contrib    = (1 - ssim_val) * 40      # more difference → more confident
    damage_contrib  = min(damage_pct, 100) * 0.4
    zones_contrib   = min(n_zones * 2, 20)
    return min(int(ssim_contrib + damage_contrib + zones_contrib), 100)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:2.4rem;'>🛰️</div>
        <div style='font-size:1.2rem; font-weight:800; background: linear-gradient(90deg,#60a5fa,#a855f7);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>ReliefVision</div>
        <div style='font-size:0.62rem; color:rgba(255,255,255,0.35); letter-spacing:0.15em;
                    text-transform:uppercase; margin-top:4px;'>v2.0 · PBL 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🛠️ Analysis Settings**")

    disaster = st.selectbox(
        "Disaster Type",
        list(DISASTER_PROFILES.keys()),
        format_func=lambda x: f"{DISASTER_PROFILES[x]['icon']}  {x}"
    )
    profile = DISASTER_PROFILES[disaster]
    st.caption(profile["description"])

    st.markdown("---")
    sensitivity = st.slider("Damage Sensitivity", 10, 90, 50,
                            help="Higher = detect smaller/subtler damage zones")
    min_zone_px = st.slider("Min Zone Size (px²)", 100, 2000, 500,
                             help="Filter out noise below this contour area")
    blend_alpha = st.slider("Overlay Opacity", 10, 90, 45,
                             help="How strongly to blend damage overlay onto image") / 100

    st.markdown("---")
    show_heatmap   = st.checkbox("Show Damage Heatmap",    value=True)
    show_contours  = st.checkbox("Show Zone Contours",     value=True)
    show_zone_tbl  = st.checkbox("Show Zone Breakdown",    value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.68rem; color:rgba(255,255,255,0.28); text-align:center; line-height:1.7;'>
        Bhumija Goswami · 2427030257<br>
        Guide: Dr. Ajay Kumar<br>
        Dept. of CSE · MUJ
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="rv-hero">
    <h1>ReliefVision</h1>
    <p>AI-powered satellite damage assessment · {profile['icon']} {disaster} mode active</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  IMAGE UPLOAD
# ─────────────────────────────────────────────
st.markdown('<div class="rv-section-title">01 — Upload Satellite Imagery</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    before = st.file_uploader("Pre-Disaster Image", type=["jpg", "jpeg", "png"],
                               label_visibility="visible")
with col_b:
    after  = st.file_uploader("Post-Disaster Image", type=["jpg", "jpeg", "png"],
                               label_visibility="visible")


# ─────────────────────────────────────────────
#  PREVIEW UPLOADED IMAGES
# ─────────────────────────────────────────────
if before and after:
    c1, c2 = st.columns(2)
    with c1:
        st.image(before, caption="Pre-Disaster", use_container_width=True)
    with c2:
        st.image(after, caption="Post-Disaster", use_container_width=True)

    st.markdown('<div class="rv-divider"></div>', unsafe_allow_html=True)

    run_btn = st.button("🔍  Run Damage Analysis", use_container_width=True)

    if run_btn:
        with st.spinner("Analysing satellite imagery…"):

            # ── Load & resize ──────────────────────────────
            SIZE = 640
            img1 = np.array(Image.open(before).convert("RGB"))
            img2 = np.array(Image.open(after ).convert("RGB"))
            img1 = cv2.resize(img1, (SIZE, SIZE))
            img2 = cv2.resize(img2, (SIZE, SIZE))

            # ── Cloud masking ──────────────────────────────
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            _, cm1 = cv2.threshold(gray1, 220, 255, cv2.THRESH_BINARY)
            _, cm2 = cv2.threshold(gray2, 220, 255, cv2.THRESH_BINARY)
            gray1[cm1 == 255] = 0
            gray2[cm2 == 255] = 0
            img1_masked = img1.copy(); img1_masked[cm1 == 255] = 0
            img2_masked = img2.copy(); img2_masked[cm2 == 255] = 0

            # ── Weighted per-channel SSIM ──────────────────
            w = profile["channel_weights"]
            score, diff = weighted_ssim(gray1, gray2, img1_masked, img2_masked, w)

            # ── Threshold & morphology ─────────────────────
            threshold_value = 255 - int(sensitivity * 2.2)
            diff_uint8 = (diff * 255).astype("uint8")
            _, thresh = cv2.threshold(diff_uint8, threshold_value, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

            # ── Contour detection ──────────────────────────
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area   = SIZE * SIZE
            changed_area = 0
            zones        = []

            output = img2.copy()
            heatmap_base = build_heatmap(1 - diff, thresh)

            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area < min_zone_px:
                    continue
                changed_area += area
                x, y, w_box, h_box = cv2.boundingRect(c)

                # Local SSIM for this zone
                zone_diff    = diff[y:y+h_box, x:x+w_box]
                local_change = float(1 - zone_diff.mean())
                local_pct    = local_change * 100
                zone_lvl     = classify_zone(local_pct)

                zones.append({
                    "id":       i + 1,
                    "x": x, "y": y, "w": w_box, "h": h_box,
                    "area_px":  int(area),
                    "area_pct": round((area / total_area) * 100, 2),
                    "severity": zone_lvl["label"],
                    "action":   zone_lvl["action"],
                    "color":    zone_lvl["color"],
                })

                if show_contours:
                    col = zone_lvl["color"]
                    cv2.rectangle(output, (x, y), (x+w_box, y+h_box), col, 2)
                    cv2.putText(output, f"Z{i+1} {zone_lvl['label']}",
                                (x+4, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                col, 1, cv2.LINE_AA)

            # ── Overlay heatmap ────────────────────────────
            if show_heatmap:
                overlay = cv2.addWeighted(output, 1 - blend_alpha,
                                           heatmap_base, blend_alpha, 0)
            else:
                overlay = output

            # ── Global metrics ─────────────────────────────
            global_pct  = (changed_area / total_area) * 100
            global_lvl  = classify_zone(global_pct)
            conf        = confidence_score(score, global_pct, len(zones))
            ts          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── RESULTS ────────────────────────────────────────
        st.markdown('<div class="rv-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="rv-section-title">02 — Analysis Results</div>', unsafe_allow_html=True)

        # Metric tiles
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="val val-blue">{score:.4f}</div>
                <div class="lbl">SSIM Score</div>
            </div>
            <div class="metric-tile">
                <div class="val val-{'red' if global_pct>50 else 'orange' if global_pct>25 else 'green'}">{global_pct:.1f}%</div>
                <div class="lbl">Damage Area</div>
            </div>
            <div class="metric-tile">
                <div class="val val-purple">{len(zones)}</div>
                <div class="lbl">Zones Detected</div>
            </div>
            <div class="metric-tile">
                <div class="val val-{'red' if conf>70 else 'orange' if conf>40 else 'green'}">{conf}%</div>
                <div class="lbl">Confidence</div>
            </div>
            <div class="metric-tile">
                <span class="sev-badge sev-{global_lvl['label']}">{global_lvl['label']}</span>
                <div class="lbl" style="margin-top:8px;">Overall Severity</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Output image
        st.markdown('<div class="rv-section-title">03 — Damage Visualisation</div>', unsafe_allow_html=True)
        st.image(overlay, caption="Damage Detection Overlay", use_container_width=True)

        # Heatmap standalone
        if show_heatmap:
            with st.expander("🌡️ View Isolated Damage Heatmap"):
                st.image(heatmap_base, caption="SSIM Difference Heatmap — Red = High Damage, Blue = No Damage",
                         use_container_width=True)

        # ── ZONE TABLE ─────────────────────────────────────
        if show_zone_tbl and zones:
            st.markdown('<div class="rv-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="rv-section-title">04 — Zone-Level Breakdown</div>', unsafe_allow_html=True)

            rows = "".join([
                f"""<tr>
                    <td>Zone {z['id']}</td>
                    <td>{z['area_px']:,} px²</td>
                    <td>{z['area_pct']}%</td>
                    <td><span class='sev-badge sev-{z['severity']}'>{z['severity']}</span></td>
                    <td style='font-size:0.82rem;'>{z['action']}</td>
                </tr>"""
                for z in sorted(zones, key=lambda x: x["area_pct"], reverse=True)
            ])

            st.markdown(f"""
            <table class="zone-table">
                <thead><tr>
                    <th>Zone</th><th>Area (px²)</th><th>% of Frame</th>
                    <th>Severity</th><th>Recommended Action</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

        # ── ALERT PANEL ────────────────────────────────────
        st.markdown('<div class="rv-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="rv-section-title">05 — Alert Summary</div>', unsafe_allow_html=True)

        action_text = global_lvl["action"]
        if global_lvl["label"] == "CRITICAL":
            st.error(f"🚨 **CRITICAL ALERT** · {disaster}\n\n{action_text}")
        elif global_lvl["label"] == "HIGH":
            st.warning(f"⚠️ **HIGH SEVERITY** · {disaster}\n\n{action_text}")
        elif global_lvl["label"] == "MEDIUM":
            st.warning(f"📡 **MEDIUM SEVERITY** · {disaster}\n\n{action_text}")
        else:
            st.success(f"✅ **LOW IMPACT** · {disaster}\n\n{action_text}")

        # ── DOWNLOADABLE REPORT ────────────────────────────
        st.markdown('<div class="rv-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="rv-section-title">06 — Export Report</div>', unsafe_allow_html=True)

        report = {
            "report_id":       f"RV-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at":    ts,
            "system":          "ReliefVision v2.0",
            "analyst":         "Bhumija Goswami · 2427030257",
            "disaster_type":   disaster,
            "channel_weights": {"B": w[0], "G": w[1], "R": w[2]},
            "sensitivity":     sensitivity,
            "ssim_score":      round(score, 6),
            "global_damage_pct": round(global_pct, 2),
            "overall_severity":  global_lvl["label"],
            "confidence_pct":    conf,
            "zones_detected":    len(zones),
            "zone_details": [
                {
                    "zone_id":  z["id"],
                    "bbox":     {"x": z["x"], "y": z["y"], "w": z["w"], "h": z["h"]},
                    "area_px":  z["area_px"],
                    "area_pct": z["area_pct"],
                    "severity": z["severity"],
                }
                for z in zones
            ],
            "recommended_action": global_lvl["action"],
        }

        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="⬇️  Download JSON Report",
            data=report_json,
            file_name=f"reliefvision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

        # Plain text version
        txt_lines = [
            "=" * 54,
            "  RELIEFVISION DAMAGE ASSESSMENT REPORT",
            "=" * 54,
            f"  Report ID    : {report['report_id']}",
            f"  Generated    : {ts}",
            f"  Disaster Type: {disaster}",
            f"  Analyst      : {report['analyst']}",
            "-" * 54,
            f"  SSIM Score   : {score:.4f}",
            f"  Damage Area  : {global_pct:.2f}%",
            f"  Severity     : {global_lvl['label']}",
            f"  Confidence   : {conf}%",
            f"  Zones Found  : {len(zones)}",
            "-" * 54,
            "  ZONE BREAKDOWN",
        ]
        for z in sorted(zones, key=lambda x: x["area_pct"], reverse=True):
            txt_lines.append(f"  Zone {z['id']:02d} | {z['severity']:8s} | {z['area_pct']}% of frame")
        txt_lines += [
            "-" * 54,
            f"  ACTION: {global_lvl['action']}",
            "=" * 54,
        ]

        st.download_button(
            label="⬇️  Download Text Report",
            data="\n".join(txt_lines),
            file_name=f"reliefvision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

elif before or after:
    st.info("Please upload **both** a pre-disaster and post-disaster image to begin analysis.")
else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 48px 32px;">
        <div style="font-size: 3rem; margin-bottom: 16px;">🛰️</div>
        <div style="font-size: 1.1rem; font-weight: 800; color:#f1f5f9; margin-bottom: 10px;">
            Ready to Analyse
        </div>
        <div style="font-size: 0.9rem; color: rgba(255,255,255,0.5); line-height: 1.7;">
            Upload a pre-disaster and post-disaster satellite image<br>
            using the panels above to begin AI damage assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)
