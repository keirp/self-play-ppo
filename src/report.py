"""PDF report generation with plots, tables, and interpretation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def smooth(data, window=5):
    if len(data) <= window:
        return data
    kernel = np.ones(window) / window
    # Use 'same' mode to preserve length
    smoothed = np.convolve(data, kernel, mode="same")
    return smoothed


def plot_training_curves(metrics, save_path, title_prefix=""):
    """Generate training curve plots and save as images."""
    plots = []

    # 1. Win/Draw/Loss vs random over training
    if "vs_random_win_rate" in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        iters = metrics["eval_iteration"]
        ax.plot(iters, metrics["vs_random_win_rate"], label="Win", color="green")
        ax.plot(iters, metrics["vs_random_draw_rate"], label="Draw", color="blue")
        ax.plot(iters, metrics["vs_random_loss_rate"], label="Loss", color="red")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Rate")
        ax.set_title(f"{title_prefix}vs Random Opponent")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        path = os.path.join(save_path, "vs_random.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(("vs_random.png", "Win/Draw/Loss rate against random opponent"))

    # 2. vs optimal
    if "vs_optimal_draw_rate" in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        iters = metrics["eval_iteration"]
        ax.plot(iters, metrics["vs_optimal_win_rate"], label="Win", color="green")
        ax.plot(iters, metrics["vs_optimal_draw_rate"], label="Draw", color="blue")
        ax.plot(iters, metrics["vs_optimal_loss_rate"], label="Loss", color="red")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Rate")
        ax.set_title(f"{title_prefix}vs Optimal (Minimax) Opponent")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        path = os.path.join(save_path, "vs_optimal.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(("vs_optimal.png", "Win/Draw/Loss rate against optimal opponent"))

    # 3. Policy/Value loss
    if "policy_loss" in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        iters = metrics["iteration"]
        ax1.plot(iters, smooth(metrics["policy_loss"], 10), color="purple")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Policy Loss")
        ax1.set_title(f"{title_prefix}Policy Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(iters, smooth(metrics["value_loss"], 10), color="orange")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Value Loss")
        ax2.set_title(f"{title_prefix}Value Loss")
        ax2.grid(True, alpha=0.3)

        path = os.path.join(save_path, "losses.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(("losses.png", "Policy and Value losses during training"))

    # 4. Entropy
    if "entropy" in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        iters = metrics["iteration"]
        ax.plot(iters, smooth(metrics["entropy"], 10), color="teal")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Entropy")
        ax.set_title(f"{title_prefix}Policy Entropy")
        ax.grid(True, alpha=0.3)
        path = os.path.join(save_path, "entropy.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(("entropy.png", "Policy entropy over training"))

    # 5. Self-play results
    if "sp_win_rate" in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        iters = metrics["iteration"]
        ax.plot(iters, smooth(metrics["sp_win_rate"], 10), label="Win", color="green")
        ax.plot(iters, smooth(metrics["sp_draw_rate"], 10), label="Draw", color="blue")
        ax.plot(iters, smooth(metrics["sp_loss_rate"], 10), label="Loss", color="red")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Rate (smoothed)")
        ax.set_title(f"{title_prefix}Self-Play Results vs Pool Opponents")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        path = os.path.join(save_path, "self_play.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(("self_play.png", "Self-play win/draw/loss against opponent pool"))

    return plots


def plot_comparison(all_metrics, labels, save_path, metric_key, ylabel, title):
    """Plot a comparison of multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for metrics, label in zip(all_metrics, labels):
        if metric_key not in metrics:
            continue
        data = metrics[metric_key]
        # Use eval_iteration for eval metrics, iteration for per-iter metrics
        if len(data) == len(metrics.get("eval_iteration", [])):
            x = metrics["eval_iteration"]
        elif len(data) == len(metrics.get("iteration", [])):
            x = metrics["iteration"]
            data = smooth(data, 10)
        else:
            continue
        ax.plot(x, data, label=label)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fname = f"comparison_{metric_key}.png"
    path = os.path.join(save_path, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fname


def generate_pdf_report(
    report_path,
    title,
    sections,
    plot_dir,
):
    """Generate a PDF report.

    sections: list of dicts with keys:
        - "heading": section title
        - "text": paragraph text
        - "plots": list of (filename, caption) tuples
        - "table": {"headers": [...], "rows": [[...], ...]} (optional)
    """
    doc = SimpleDocTemplate(report_path, pagesize=letter,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"], fontSize=20, spaceAfter=20
    )
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))

    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"], fontSize=14, spaceAfter=8, spaceBefore=16
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=9, textColor=colors.grey,
        alignment=1, spaceAfter=12
    )

    for section in sections:
        if "heading" in section:
            story.append(Paragraph(section["heading"], heading_style))

        if "text" in section:
            for para in section["text"].split("\n\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(para, body_style))

        if "table" in section:
            t = section["table"]
            data = [t["headers"]] + t["rows"]
            table = Table(data, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#D9E2F3")]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(Spacer(1, 8))
            story.append(table)
            story.append(Spacer(1, 8))

        if "plots" in section:
            for fname, caption in section["plots"]:
                img_path = os.path.join(plot_dir, fname)
                if os.path.exists(img_path):
                    img = Image(img_path, width=6.5*inch, height=3.25*inch)
                    story.append(img)
                    story.append(Paragraph(caption, caption_style))

        if section.get("page_break"):
            story.append(PageBreak())

    doc.build(story)
    print(f"Report saved to: {report_path}", flush=True)
