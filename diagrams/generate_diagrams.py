"""
Generates UML class and sequence diagrams as PNG files.
Run:  python diagrams/generate_diagrams.py
Outputs: diagrams/class_diagram.png, diagrams/sequence_diagram.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_class_box(ax, x, y, w, name, attrs, methods,
                   is_interface=False, color="#D6EAF8"):
    """
    Draw a UML-style class box with three horizontal compartments.
    Returns the (center_x, bottom_y, top_y) of the box.
    """
    row_h = 0.30
    name_h = 0.55 if not is_interface else 0.70
    attr_h = max(0.30, len(attrs) * row_h + 0.10)
    meth_h = max(0.30, len(methods) * row_h + 0.10)
    total_h = name_h + attr_h + meth_h

    top = y
    bottom = y - total_h

    rect = mpatches.FancyBboxPatch(
        (x, bottom), w, total_h,
        boxstyle="square,pad=0",
        linewidth=1.4,
        edgecolor="#2C3E50",
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(rect)

    # Dividers
    ax.plot([x, x+w], [top - name_h, top - name_h],
            color="#2C3E50", lw=0.8, zorder=3)
    ax.plot([x, x+w], [top - name_h - attr_h, top - name_h - attr_h],
            color="#2C3E50", lw=0.8, zorder=3)

    # Stereotype + name
    ty = top - 0.10
    if is_interface:
        ax.text(x + w/2, ty, "«interface»", ha="center", va="top",
                fontsize=7, style="italic", color="#555555", zorder=4)
        ty -= 0.22
    ax.text(x + w/2, ty, name, ha="center", va="top",
            fontsize=8.5, fontweight="bold", color="#1A252F", zorder=4)

    # Attributes
    for i, attr in enumerate(attrs):
        ax.text(x + 0.07, top - name_h - 0.08 - i * row_h, attr,
                ha="left", va="top", fontsize=6.5, color="#1A252F",
                fontfamily="monospace", zorder=4)

    # Methods
    for i, meth in enumerate(methods):
        ax.text(x + 0.07, top - name_h - attr_h - 0.08 - i * row_h, meth,
                ha="left", va="top", fontsize=6.5, color="#1A252F",
                fontfamily="monospace", zorder=4)

    cx = x + w / 2
    return cx, bottom, top


def arrow(ax, x1, y1, x2, y2, style="->", color="#2C3E50", lw=1.2,
          label="", dashed=False):
    ls = "--" if dashed else "-"
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.05, my, label, fontsize=6, color="#555555", zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# Class Diagram
# ─────────────────────────────────────────────────────────────────────────────

def make_class_diagram():
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    ax.set_title("RAG Printer Helpdesk – UML Class Diagram (Project 2)",
                 fontsize=13, fontweight="bold", pad=10, color="#1A252F")

    W = 3.5          # standard box width
    IFACE_W = 3.0    # interface box width
    IC = "#EBF5FB"   # interface fill
    CC = "#D5F5E3"   # concrete class fill
    OC = "#FEF9E7"   # orchestrator fill
    SC = "#F9EBEA"   # support fill

    boxes = {}

    # Row 1 – Interfaces  (y_top = 15.5)
    boxes["IEmbedder"] = draw_class_box(
        ax, 0.5, 15.5, IFACE_W, "IEmbedder",
        [], ["embed(text) : List[float]", "embed_batch(texts) : List"],
        is_interface=True, color=IC)

    boxes["IVectorStore"] = draw_class_box(
        ax, 5.5, 15.5, IFACE_W+0.5, "IVectorStore",
        [], ["index_documents(chunks)", "search(query,top_k) : List", "count() : int"],
        is_interface=True, color=IC)

    boxes["ILLMGenerator"] = draw_class_box(
        ax, 12.0, 15.5, IFACE_W, "ILLMGenerator",
        [], ["generate(prompt, **kw) : str"],
        is_interface=True, color=IC)

    # Row 2 – Concrete implementations (y_top = 11.8)
    boxes["EmbeddingModel"] = draw_class_box(
        ax, 0.5, 11.8, W, "EmbeddingModel",
        ["model_name: str", "model: SentenceTransformer"],
        ["embed(text)", "embed_batch(texts)"],
        color=CC)

    boxes["VectorStore"] = draw_class_box(
        ax, 5.2, 11.8, W+0.3, "VectorStore",
        ["embedder: IEmbedder", "collection_name: str"],
        ["index_documents(chunks)", "search(query,top_k)", "count()"],
        color=CC)

    boxes["LLMGenerator"] = draw_class_box(
        ax, 12.0, 11.8, W, "LLMGenerator",
        ["model_name: str", "generator: pipeline"],
        ["generate(prompt, **kw)"],
        color=CC)

    # Row 3 – Orchestrators (y_top = 8.0)
    boxes["RAGSystem"] = draw_class_box(
        ax, 1.5, 8.0, W+0.5, "RAGSystem",
        ["vector_store: IVectorStore"],
        ["retrieve(query,top_k)", "format_context(chunks)",
         "_initialize_index()"],
        color=OC)

    boxes["RAGGenerator"] = draw_class_box(
        ax, 10.0, 8.0, W+0.5, "RAGGenerator",
        ["llm: ILLMGenerator", "prompt_builder: PromptBuilder"],
        ["generate_answer(ctx,query,cites)"],
        color=OC)

    # Row 3 right – helpers
    boxes["PromptBuilder"] = draw_class_box(
        ax, 14.5, 8.0, W, "PromptBuilder",
        [],
        ["build_retrieval_prompt(ctx,q)"],
        color=SC)

    boxes["AnswerProcessor"] = draw_class_box(
        ax, 14.5, 5.6, W, "AnswerProcessor",
        ["UNCERTAINTY_PHRASES: List"],
        ["clean_answer(text)", "contains_uncertainty(text)",
         "process(answer) : Tuple"],
        color=SC)

    # Row 4 – data ingest + CLI (y_top = 5.0)
    boxes["TextChunker"] = draw_class_box(
        ax, 0.3, 5.0, W, "TextChunker",
        ["chunk_size: int", "chunk_overlap: int"],
        ["chunk_text(text, meta)", "split_into_sentences(text)",
         "estimate_tokens(text)"],
        color=SC)

    boxes["DocumentLoader"] = draw_class_box(
        ax, 4.3, 5.0, W, "DocumentLoader",
        ["data_dir: Path", "chunker: TextChunker"],
        ["load_documents()"],
        color=SC)

    boxes["CLIInterface"] = draw_class_box(
        ax, 8.0, 5.0, W+0.3, "CLIInterface",
        ["rag_system: RAGSystem", "generator: RAGGenerator",
         "query_history: List"],
        ["initialize()", "process_query(query)",
         "handle_command(input)", "format_answer(ans,cites,conf)"],
        color=OC)

    # ── Relationships ─────────────────────────────────────────────────────────
    # EmbeddingModel --|> IEmbedder  (open hollow triangle = realisation, dashed)
    cx_em, bot_em, top_em = boxes["EmbeddingModel"]
    cx_ie, bot_ie, top_ie = boxes["IEmbedder"]
    arrow(ax, cx_em, top_em, cx_ie, bot_ie, style="-|>",
          dashed=True, color="#1A5276")

    # VectorStore --|> IVectorStore
    cx_vs, bot_vs, top_vs = boxes["VectorStore"]
    cx_ivs, bot_ivs, top_ivs = boxes["IVectorStore"]
    arrow(ax, cx_vs, top_vs, cx_ivs + 0.1, bot_ivs, style="-|>",
          dashed=True, color="#1A5276")

    # LLMGenerator --|> ILLMGenerator
    cx_llm, bot_llm, top_llm = boxes["LLMGenerator"]
    cx_illm, bot_illm, top_illm = boxes["ILLMGenerator"]
    arrow(ax, cx_llm, top_llm, cx_illm, bot_illm, style="-|>",
          dashed=True, color="#1A5276")

    # RAGSystem ..> IVectorStore  (dashed dependency)
    cx_rs, bot_rs, top_rs = boxes["RAGSystem"]
    arrow(ax, cx_rs + 0.8, top_rs, cx_ivs + 0.3, bot_ivs,
          style="->", dashed=True, color="#7D6608",
          label="«uses»")

    # VectorStore ..> IEmbedder  (dashed dependency)
    arrow(ax, cx_vs - 0.2, top_vs, cx_ie + 0.2, bot_ie,
          style="->", dashed=True, color="#7D6608",
          label="«uses»")

    # RAGGenerator ..> ILLMGenerator
    cx_rg, bot_rg, top_rg = boxes["RAGGenerator"]
    arrow(ax, cx_rg + 0.5, top_rg, cx_illm - 0.2, bot_illm,
          style="->", dashed=True, color="#7D6608",
          label="«uses»")

    # RAGGenerator --> PromptBuilder  (composition)
    cx_pb, bot_pb, top_pb = boxes["PromptBuilder"]
    arrow(ax, cx_rg + 1.0, top_rg - 0.5, cx_pb, top_pb - 0.5,
          style="->", color="#117A65",
          label="uses")

    # RAGGenerator --> AnswerProcessor
    cx_ap, bot_ap, top_ap = boxes["AnswerProcessor"]
    arrow(ax, cx_rg + 1.0, top_rg - 1.0, cx_ap, top_ap - 0.5,
          style="->", color="#117A65",
          label="uses")

    # DocumentLoader --> TextChunker
    cx_dl, bot_dl, top_dl = boxes["DocumentLoader"]
    cx_tc, bot_tc, top_tc = boxes["TextChunker"]
    arrow(ax, cx_dl - 0.5, top_dl - 0.3, cx_tc + 1.5, top_tc - 0.3,
          style="->", color="#117A65")

    # RAGSystem --> DocumentLoader  (uses in _initialize_index)
    arrow(ax, cx_rs + 0.2, bot_rs, cx_dl, top_dl,
          style="->", color="#117A65", label="uses")

    # CLIInterface --> RAGSystem
    cx_cli, bot_cli, top_cli = boxes["CLIInterface"]
    arrow(ax, cx_cli - 1.0, top_cli - 0.4, cx_rs + 1.5, top_rs - 0.4,
          style="->", color="#7D3C98", label="uses")

    # CLIInterface --> RAGGenerator
    arrow(ax, cx_cli + 0.3, top_cli - 0.4, cx_rg - 0.8, top_rg - 0.4,
          style="->", color="#7D3C98", label="uses")

    # Legend
    legend_x, legend_y = 17.0, 4.8
    ax.text(legend_x, legend_y, "Legend", fontsize=8, fontweight="bold",
            color="#1A252F")
    for i, (clr, lbl, ls) in enumerate([
        ("#1A5276", "Realises interface", "--"),
        ("#7D6608", "Depends on (DIP)", "--"),
        ("#117A65", "Uses / Composition", "-"),
        ("#7D3C98", "Uses (CLI)", "-"),
    ]):
        y_ = legend_y - 0.45 * (i+1)
        ax.annotate("",
                    xy=(legend_x + 1.4, y_), xytext=(legend_x + 0.1, y_),
                    arrowprops=dict(arrowstyle="->", color=clr, lw=1.2,
                                   linestyle=ls), zorder=5)
        ax.text(legend_x + 1.5, y_, lbl, va="center", fontsize=7,
                color="#333333")

    plt.tight_layout(pad=1.5)
    out = os.path.join(OUT_DIR, "class_diagram.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Diagram
# ─────────────────────────────────────────────────────────────────────────────

def make_sequence_diagram():
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    ax.set_title("RAG Printer Helpdesk – UML Sequence Diagram",
                 fontsize=12, fontweight="bold", pad=8, color="#1A252F")

    ACTORS = [
        ("User",           1.2),
        ("CLIInterface",   3.5),
        ("RAGSystem",      6.0),
        ("IVectorStore",   8.5),
        ("IEmbedder",     11.0),
        ("RAGGenerator",  13.5),
        ("ILLMGenerator", 16.0),
    ]

    TOP = 13.0
    BOX_H = 0.45
    BOX_W = 1.5
    COLORS = {
        "User":          "#D6EAF8",
        "CLIInterface":  "#D5F5E3",
        "RAGSystem":     "#FEF9E7",
        "IVectorStore":  "#EBF5FB",
        "IEmbedder":     "#EBF5FB",
        "RAGGenerator":  "#FDEDEC",
        "ILLMGenerator": "#EBF5FB",
    }

    # Draw actor headers and lifelines
    lifeline_x = {}
    for name, x in ACTORS:
        lifeline_x[name] = x
        rect = mpatches.FancyBboxPatch(
            (x - BOX_W/2, TOP), BOX_W, BOX_H,
            boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="#2C3E50",
            facecolor=COLORS[name], zorder=3)
        ax.add_patch(rect)
        ax.text(x, TOP + BOX_H/2, name, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#1A252F", zorder=4)
        ax.plot([x, x], [0.3, TOP], color="#AAAAAA", lw=0.8,
                linestyle="--", zorder=1)

    def msg(frm, to, y, label, ret=False, color="#2C3E50"):
        x1 = lifeline_x[frm]
        x2 = lifeline_x[to]
        ls = "--" if ret else "-"
        style = "->" if not ret else "->"
        ax.annotate(
            "",
            xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(
                arrowstyle=style,
                color=color,
                lw=1.1,
                linestyle=ls,
            ),
            zorder=5,
        )
        mx = (x1 + x2) / 2
        va = "bottom" if not ret else "top"
        ax.text(mx, y + (0.06 if not ret else -0.06), label,
                ha="center", va=va, fontsize=6.8, color=color, zorder=6)

    step = 0.95
    y = TOP - 0.1

    steps = [
        # (from,         to,              label,                                                 is_return, color)
        ("User",         "CLIInterface",  "1. input(query)",                                     False, "#1A5276"),
        ("CLIInterface", "RAGSystem",     "2. retrieve(query, top_k=5)",                          False, "#117A65"),
        ("RAGSystem",    "IVectorStore",  "3. search(query, top_k)",                              False, "#117A65"),
        ("IVectorStore", "IEmbedder",     "4. embed(query)",                                      False, "#117A65"),
        ("IEmbedder",    "IVectorStore",  "5. return embedding",                                  True,  "#888888"),
        ("IVectorStore", "RAGSystem",     "6. return [(text, meta, score)]",                      True,  "#888888"),
        ("RAGSystem",    "CLIInterface",  "7. format_context(retrieved) → (context, citations)",  False, "#117A65"),
        ("CLIInterface", "RAGGenerator",  "8. generate_answer(context, query, citations)",        False, "#7D3C98"),
        ("RAGGenerator", "RAGGenerator",  "9. build_retrieval_prompt(context, query)",            False, "#C0392B"),
        ("RAGGenerator", "ILLMGenerator", "10. generate(prompt)",                                 False, "#7D6608"),
        ("ILLMGenerator","RAGGenerator",  "11. return raw_answer",                                True,  "#888888"),
        ("RAGGenerator", "RAGGenerator",  "12. AnswerProcessor.process(raw_answer)",              False, "#C0392B"),
        ("RAGGenerator", "CLIInterface",  "13. return (answer, citations, is_confident)",         True,  "#888888"),
        ("CLIInterface", "CLIInterface",  "14. format_answer(answer, citations, is_confident)",   False, "#C0392B"),
        ("CLIInterface", "User",          "15. display formatted output",                         True,  "#1A5276"),
    ]

    for (frm, to, label, ret, color) in steps:
        y -= step
        if frm == to:
            # Self-call: draw small loop
            x_ = lifeline_x[frm]
            ax.annotate("",
                        xy=(x_, y), xytext=(x_ + 0.5, y + 0.3),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
                        zorder=5)
            ax.plot([x_, x_+0.5, x_+0.5], [y+0.3, y+0.3, y],
                    color=color, lw=1.0, zorder=4)
            ax.text(x_ + 0.55, y + 0.15, label, ha="left", va="center",
                    fontsize=6.8, color=color, zorder=6)
        else:
            msg(frm, to, y, label, ret=ret, color=color)

    # Footer
    ax.text(9, 0.15,
            "Dashed return arrows  |  All external calls go through interfaces (DIP)",
            ha="center", va="center", fontsize=7, color="#555555",
            style="italic")

    plt.tight_layout(pad=1.0)
    out = os.path.join(OUT_DIR, "sequence_diagram.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("Generating UML diagrams...")
    make_class_diagram()
    make_sequence_diagram()
    print("Done.")
