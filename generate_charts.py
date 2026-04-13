"""
generate_charts.py
Generates all README charts and saves them to assets/
Run once before pushing to GitHub.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORS = {
    "Rational Accountant": "#378ADD",
    "Selfless Cooperator": "#1D9E75",
    "Greedy Infiltrator":  "#D85A30",
    "gemma4:e2b":          "#7F77DD",
    "llama3.1:8b":         "#378ADD",
    "dolphin-mistral:7b":  "#D85A30",
}

plt.rcParams.update({
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.facecolor": "#0d1117",   # GitHub dark bg
    "axes.facecolor":   "#0d1117",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "axes.edgecolor":   "#30363d",
    "grid.color":       "#21262d",
})


# ── CHART 1: Death Spiral ────────────────────────────────────────────────────
def chart_death_spiral():
    rounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    data = {
        "Rational Accountant": [10, 10.50, 11.00, 10.50, 8.50, 2.50, 0.00, 0.50, 0.50],
        "Selfless Cooperator": [10,  6.50,  3.00,  1.50, 0.50, 0.00, 1.50, 0.00, 0.00],
        "Greedy Infiltrator":  [10,  6.50,  3.00,  1.50, 0.00, 1.00, 0.50, 0.00, 0.00],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")

    styles = {
        "Rational Accountant": {"ls": "-",  "marker": "o"},
        "Selfless Cooperator": {"ls": "--", "marker": "s"},
        "Greedy Infiltrator":  {"ls": ":",  "marker": "^"},
    }

    for agent, wealth in data.items():
        ax.plot(rounds, wealth,
                color=COLORS[agent],
                linewidth=2.5,
                label=agent,
                marker=styles[agent]["marker"],
                markersize=7,
                linestyle=styles[agent]["ls"])

    # Annotate collapse
    ax.annotate("Total collapse\n0.50 gold remaining",
                xy=(8, 0.50), xytext=(6.2, 6),
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.2),
                fontsize=9, color="#8b949e")

    ax.annotate("Started: 30 gold\nEnded: 0.50 gold",
                xy=(0, 10), xytext=(0.2, 11.5),
                fontsize=8.5, color="#8b949e",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d",
                          edgecolor="#30363d", alpha=0.9))

    ax.fill_between(rounds, data["Selfless Cooperator"],
                    alpha=0.07, color=COLORS["Selfless Cooperator"])
    ax.fill_between(rounds, data["Greedy Infiltrator"],
                    alpha=0.07, color=COLORS["Greedy Infiltrator"])

    ax.set_xlim(-0.2, 8.3)
    ax.set_ylim(-0.5, 14)
    ax.set_xticks(rounds)
    ax.set_xticklabels(["Start"] + [f"R{i}" for i in range(1, 9)], fontsize=10)
    ax.set_ylabel("Wealth (gold)", fontsize=11)
    ax.set_title("Experiment 2 — The Death Spiral\nSanctions destroyed 98% of group wealth in 8 rounds",
                 fontsize=12, pad=14, color="#e6edf3")

    leg = ax.legend(loc="upper right", framealpha=0.15,
                    facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#c9d1d9", fontsize=9)

    plt.tight_layout()
    plt.savefig("assets/chart_death_spiral.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Saved: chart_death_spiral.png")


# ── CHART 2: Sanction Classification ─────────────────────────────────────────
def chart_sanction_types():
    categories   = ["Prosocial\n(punished defector)",
                    "Antisocial\n(punished cooperator)",
                    "Irrational\n(punished bankrupt)"]
    counts       = [6, 6, 1]
    bar_colors   = ["#1D9E75", "#D85A30", "#EF9F27"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#0d1117")

    bars = ax.bar(categories, counts, color=bar_colors,
                  width=0.5, zorder=3, edgecolor="#0d1117", linewidth=1.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count), ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#e6edf3")

    ax.set_ylim(0, 9)
    ax.set_ylabel("Number of sanctions", fontsize=11)
    ax.set_title("Experiment 2 — Sanction Types\nAntisocial punishment matched prosocial punishment exactly",
                 fontsize=12, pad=14, color="#e6edf3")
    ax.set_yticks(range(0, 9))

    plt.tight_layout()
    plt.savefig("assets/chart_sanction_types.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Saved: chart_sanction_types.png")


# ── CHART 3: Context vs Stripped std dev ─────────────────────────────────────
def chart_context_stability():
    agents   = ["Selfless\nCooperator", "Rational\nAccountant", "Greedy\nInfiltrator"]
    full_sd  = [0.52, 0.41, 0.75]
    strip_sd = [0.98, 0.64, 0.67]

    x     = np.arange(len(agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")

    b1 = ax.bar(x - width/2, full_sd,  width, label="Full context",
                color="#378ADD", alpha=0.9, zorder=3)
    b2 = ax.bar(x + width/2, strip_sd, width, label="Stripped (no history)",
                color="#8b949e", alpha=0.9, zorder=3)

    ax.set_ylabel("Contribution std dev", fontsize=11)
    ax.set_title("Experiment 1 — Does Game History Stabilize Behavior?\nLower std dev = more consistent contributions",
                 fontsize=12, pad=14, color="#e6edf3")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=11)
    ax.set_ylim(0, 1.3)

    ax.annotate("47% drop in variance\nwith game history",
                xy=(-0.175, 0.52), xytext=(0.5, 1.1),
                arrowprops=dict(arrowstyle="->", color="#1D9E75", lw=1.2),
                fontsize=8.5, color="#1D9E75")

    leg = ax.legend(framealpha=0.15, facecolor="#161b22",
                    edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

    plt.tight_layout()
    plt.savefig("assets/chart_context_stability.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Saved: chart_context_stability.png")


# ── CHART 4: Persona vs Model range ──────────────────────────────────────────
def chart_persona_vs_model():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")

    # Left: by persona
    personas = ["Selfless\nCooperator", "Rational\nAccountant", "Greedy\nInfiltrator"]
    p_avgs   = [4.21, 2.35, 0.92]
    p_colors = [COLORS["Selfless Cooperator"],
                COLORS["Rational Accountant"],
                COLORS["Greedy Infiltrator"]]

    bars1 = ax1.bar(personas, p_avgs, color=p_colors, width=0.5,
                    zorder=3, edgecolor="#0d1117", linewidth=1.5)
    for bar, val in zip(bars1, p_avgs):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.07,
                 f"{val:.2f}", ha="center", va="bottom",
                 fontsize=11, color="#e6edf3")

    ax1.set_ylim(0, 5.5)
    ax1.set_ylabel("Avg contribution (0-5 scale)", fontsize=10)
    ax1.set_title("By Persona Label\nRange: 3.29", fontsize=11,
                  pad=10, color="#e6edf3")
    ax1.text(0.5, 5.2, "← DOMINANT DRIVER →",
             ha="center", fontsize=9, color="#1D9E75",
             transform=ax1.transData)

    # Right: by model
    models  = ["gemma4:e2b", "llama3.1:8b", "dolphin-\nmistral:7b"]
    m_avgs  = [2.22, 2.83, 2.42]
    m_color = "#8b949e"

    bars2 = ax2.bar(models, m_avgs, color=m_color, width=0.5,
                    zorder=3, edgecolor="#0d1117", linewidth=1.5, alpha=0.7)
    for bar, val in zip(bars2, m_avgs):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.07,
                 f"{val:.2f}", ha="center", va="bottom",
                 fontsize=11, color="#e6edf3")

    ax2.set_ylim(0, 5.5)
    ax2.set_title("By Model\nRange: 0.61", fontsize=11,
                  pad=10, color="#e6edf3")
    ax2.text(0.5, 5.2, "← CLUSTERED, NO PATTERN →",
             ha="center", fontsize=9, color="#8b949e",
             transform=ax2.transData)

    fig.suptitle("Experiment 5 — Persona Label vs Model Weights\nWhat actually drives agent behavior?",
                 fontsize=13, y=1.02, color="#e6edf3")

    plt.tight_layout()
    plt.savefig("assets/chart_persona_vs_model.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Saved: chart_persona_vs_model.png")


# ── CHART 5: Communication pool history ──────────────────────────────────────
def chart_communication_pool():
    rounds     = [1, 2, 3, 4, 5, 6, 7, 8]
    exp2_pool  = [11, 11, 9, 6, 4, 3, 1, 0]
    exp4_pool  = [10, 12, 9, 10, 7, 4, 4, 2]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")

    ax.plot(rounds, exp2_pool, color="#D85A30", linewidth=2.5,
            label="Sanctions only (Exp 2)", marker="o", markersize=7)
    ax.plot(rounds, exp4_pool, color="#1D9E75", linewidth=2.5,
            label="Communication + Sanctions (Exp 4)",
            marker="s", markersize=7, linestyle="--")

    ax.fill_between(rounds, exp2_pool, exp4_pool,
                    where=[e4 >= e2 for e4, e2 in zip(exp4_pool, exp2_pool)],
                    alpha=0.12, color="#1D9E75",
                    label="Cooperation gain from communication")

    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(-0.5, 15)
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"R{i}" for i in rounds], fontsize=10)
    ax.set_ylabel("Pool total (gold contributed)", fontsize=11)
    ax.set_title("Experiment 4 — Did Communication Help?\nHigher pool = more cooperation that round",
                 fontsize=12, pad=14, color="#e6edf3")

    leg = ax.legend(loc="upper right", framealpha=0.15,
                    facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#c9d1d9", fontsize=9)

    plt.tight_layout()
    plt.savefig("assets/chart_communication_pool.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Saved: chart_communication_pool.png")


# ── CHART 6: Instruction compliance ──────────────────────────────────────────
def chart_compliance():
    conditions = ["Single instruction\n(Exp 2)", "Triple-redundant\n(Exp 3)"]
    compliance = [0, 100]
    bar_colors = ["#D85A30", "#1D9E75"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0d1117")

    bars = ax.bar(conditions, compliance, color=bar_colors,
                  width=0.4, zorder=3, edgecolor="#0d1117", linewidth=1.5)

    for bar, val in zip(bars, compliance):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f"{val}%", ha="center", va="bottom",
                fontsize=18, fontweight="bold", color="#e6edf3")

    ax.set_ylim(0, 120)
    ax.set_ylabel("Compliance rate (%)", fontsize=11)
    ax.set_title('Experiment 3 — "Never Sanction" Instruction\nSame model, same game, different prompt structure',
                 fontsize=12, pad=14, color="#e6edf3")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))

    plt.tight_layout()
    plt.savefig("assets/chart_compliance.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Saved: chart_compliance.png")


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)
    print("Generating charts...")
    chart_death_spiral()
    chart_sanction_types()
    chart_context_stability()
    chart_persona_vs_model()
    chart_communication_pool()
    chart_compliance()
    print("\nAll charts saved to assets/")
