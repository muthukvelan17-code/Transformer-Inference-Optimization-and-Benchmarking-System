"""
Report generation for benchmark results: CSV, JSON, and professional PDF export.
"""
import io
import json
from typing import Any, Dict, List, Optional
import pandas as pd

from benchmark.device_info import get_system_hardware_info


def generate_csv_report(runs: List[Dict[str, Any]]) -> str:
    """
    Export benchmark experiment history as a clean CSV string.
    """
    if not runs:
        return "No benchmark runs recorded."

    # Define clean column selection and renaming
    export_rows = []
    for r in runs:
        row = {
            "Experiment ID": r.get("experiment_id"),
            "Timestamp": r.get("timestamp"),
            "Model": r.get("model_name"),
            "Device": r.get("device"),
            "Precision": r.get("precision"),
            "Batch Size": r.get("batch_size"),
            "Seq Length": r.get("max_seq_length"),
            "Status": r.get("status"),
            "Mean Latency (ms)": r.get("mean_latency_ms", "N/A"),
            "Median Latency (ms)": r.get("median_latency_ms", "N/A"),
            "p95 Latency (ms)": r.get("p95_latency_ms", "N/A"),
            "p99 Latency (ms)": r.get("p99_latency_ms", "N/A"),
            "Throughput (samples/s)": r.get("throughput_samples_per_sec", "N/A"),
            "Total Time (s)": r.get("total_inference_time_sec", "N/A"),
            "Warmup Time (s)": r.get("warmup_time_sec", "N/A"),
            "RAM RSS (MB)": r.get("memory_rss_mb", "N/A"),
            "GPU Peak Memory (MB)": r.get("gpu_memory_peak_mb", "N/A"),
            "Warmup Iters": r.get("warmup_iterations"),
            "Bench Iters": r.get("benchmark_iterations"),
            "Error Details": r.get("error_message", ""),
        }
        export_rows.append(row)

    df = pd.DataFrame(export_rows)
    return df.to_csv(index=False)


def generate_json_report(
    system_info: Optional[Dict[str, Any]],
    runs: List[Dict[str, Any]],
    comparisons: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate comprehensive JSON report containing telemetry, configuration, and run history.
    """
    sys_info = system_info or get_system_hardware_info()

    payload = {
        "report_title": "Transformer Inference Optimization and Benchmarking System Report",
        "system_environment": sys_info,
        "experiments_count": len(runs),
        "experiments": runs,
        "comparisons": comparisons or {},
        "notes": (
            "Measurements represent real inference forward passes executed via torch.inference_mode(). "
            "FP16 requires a CUDA GPU. INT8 is benchmarked using PyTorch Dynamic Quantization on CPU."
        ),
    }

    return json.dumps(payload, indent=2, default=str)


def generate_pdf_report(
    system_info: Optional[Dict[str, Any]],
    runs: List[Dict[str, Any]],
    comparisons: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate an executive PDF benchmark report using reportlab.
    Returns bytes buffer.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        raise ImportError(
            "The 'reportlab' package is required for PDF generation. Install with `pip install reportlab`."
        )

    sys_info = system_info or get_system_hardware_info()
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
    )
    cell_style = ParagraphStyle(
        "CellCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    cell_header = ParagraphStyle(
        "CellHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("Transformer Inference Benchmark Report", title_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("Empirical Latency, Throughput & Optimization Analysis", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceBefore=8, spaceAfter=12))

    # Section 1: System Environment
    elements.append(Paragraph("1. Hardware & System Environment", h2_style))
    env_data = [
        [
            Paragraph("<b>OS:</b>", cell_style), Paragraph(str(sys_info.get("os", "N/A")), cell_style),
            Paragraph("<b>Python:</b>", cell_style), Paragraph(str(sys_info.get("python_version", "N/A")), cell_style),
        ],
        [
            Paragraph("<b>PyTorch:</b>", cell_style), Paragraph(str(sys_info.get("torch_version", "N/A")), cell_style),
            Paragraph("<b>Device:</b>", cell_style), Paragraph(str(sys_info.get("gpu_name", "CPU")), cell_style),
        ],
        [
            Paragraph("<b>Total RAM:</b>", cell_style), Paragraph(f"{sys_info.get('ram_total_gb', 'N/A')} GB", cell_style),
            Paragraph("<b>CUDA Cores:</b>", cell_style), Paragraph("Available" if sys_info.get("cuda_available") else "None (CPU Mode)", cell_style),
        ],
    ]
    t_env = Table(env_data, colWidths=[80, 190, 80, 190])
    t_env.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_env)
    elements.append(Spacer(1, 10))

    # Section 2: Benchmark Results Table
    elements.append(Paragraph("2. Empirical Benchmark Measurements", h2_style))

    if runs:
        headers = ["Exp ID", "Prec", "Batch", "SeqLen", "Status", "Mean Lat (ms)", "p95 (ms)", "Thr (s/s)", "RAM (MB)"]
        table_rows = [[Paragraph(h, cell_header) for h in headers]]

        for r in runs[:15]:  # limit to top 15 for PDF layout
            status_text = r.get("status", "N/A")
            is_success = status_text == "SUCCESS"
            table_rows.append([
                Paragraph(str(r.get("experiment_id", ""))[-12:], cell_style),
                Paragraph(str(r.get("precision", "N/A")), cell_style),
                Paragraph(str(r.get("batch_size", "N/A")), cell_style),
                Paragraph(str(r.get("max_seq_length", "N/A")), cell_style),
                Paragraph(status_text, cell_style),
                Paragraph(f"{r.get('mean_latency_ms', 0.0):.1f}" if is_success else "N/A", cell_style),
                Paragraph(f"{r.get('p95_latency_ms', 0.0):.1f}" if is_success else "N/A", cell_style),
                Paragraph(f"{r.get('throughput_samples_per_sec', 0.0):.1f}" if is_success else "N/A", cell_style),
                Paragraph(f"{r.get('memory_rss_mb', 0.0):.1f}" if is_success else "N/A", cell_style),
            ])

        t_results = Table(table_rows, colWidths=[65, 45, 40, 45, 55, 75, 65, 75, 75])
        t_results.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_results)
    else:
        elements.append(Paragraph("No benchmark experiments were executed.", body_style))

    elements.append(Spacer(1, 10))

    # Section 3: Optimization Comparisons
    if comparisons:
        elements.append(Paragraph("3. Optimization Gains vs. Baseline", h2_style))
        comp_headers = ["Comparison", "Baseline Lat", "Optimized Lat", "Latency Gain (%)", "Throughput Gain (%)"]
        comp_rows = [[Paragraph(h, cell_header) for h in comp_headers]]

        for name, comp in comparisons.items():
            lat_gain = comp.get("latency_improvement_pct")
            thr_gain = comp.get("throughput_improvement_pct")
            comp_rows.append([
                Paragraph(name.replace("_", " "), cell_style),
                Paragraph(f"{comp.get('baseline_latency_ms', 0.0):.1f} ms", cell_style),
                Paragraph(f"{comp.get('optimized_latency_ms', 0.0):.1f} ms", cell_style),
                Paragraph(f"{lat_gain:+.1f}%" if lat_gain is not None else "N/A", cell_style),
                Paragraph(f"{thr_gain:+.1f}%" if thr_gain is not None else "N/A", cell_style),
            ])

        t_comp = Table(comp_rows, colWidths=[120, 100, 100, 110, 110])
        t_comp.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0369a1")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f9ff")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_comp)
        elements.append(Spacer(1, 10))

    # Section 4: Hardware & Optimization Notes
    elements.append(Paragraph("4. Technical Considerations & Integrity Notes", h2_style))
    notes_text = (
        "• <b>Honest Capability Checks:</b> Benchmarks only report successful runs for optimizations "
        "supported by the physical device. FP16 requires a CUDA GPU. PyTorch Dynamic Quantization (INT8) "
        "accelerates linear layers on CPU.<br/>"
        "• <b>Warmup Isolation:</b> Initial inference warmup passes are excluded from steady-state statistics "
        "to prevent JIT compilation and memory allocation noise.<br/>"
        "• <b>Numerical Fidelity:</b> Lower precision models may exhibit slight drift in output logit probabilities, "
        "which is tracked via prediction fidelity metrics."
    )
    elements.append(Paragraph(notes_text, body_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
