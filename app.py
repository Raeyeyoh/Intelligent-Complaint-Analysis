

import sys
import gradio as gr
sys.path.append('./src')

from rag_pipeline import RAGPipeline

PRODUCT_CHOICES = [
    "All Products",
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
]

EXAMPLE_QUESTIONS = [
    "Why are customers unhappy with their credit cards?",
    "What are the most common issues with money transfers?",
    "Are there complaints about unauthorised account access?",
    "What billing problems do credit card customers report?",
    "Why do personal loan customers complain about interest rates?",
    "What fraud-related complaints exist across all products?",
]

print("Loading RAG pipeline — this may take a moment...")
rag = RAGPipeline(
    vector_store_dir='./vector_store',
    generator_model='google/flan-t5-large',
    k=5,
    max_new_tokens=300,
    device=-1,
)
print("Pipeline ready.\n")


def answer_question(
    question: str,
    product_filter: str,
    history: list,
) -> tuple:
    
    if not question or not question.strip():
        history.append([question, "Please enter a question."])
        return history, "No sources — empty question.", ""

    filter_val = None if product_filter == "All Products" else product_filter

    try:
        result = rag.ask(question=question, product_filter=filter_val)
        answer = result['answer']
        sources = result['sources']

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        history.append([question, error_msg])
        return history, "Error retrieving sources.", ""

    history.append([question, answer])

    sources_md = format_sources(sources)

    return history, sources_md, ""


def format_sources(sources: list) -> str:
    """
    Format retrieved source chunks as a readable markdown string
    for display in the Sources panel.

    Design choice: Sources are always shown alongside the answer so
    users can verify claims, build trust in the system, and trace
    responses back to real complaints — critical for compliance teams.

    Args:
        sources: List of chunk dicts from retrieve().

    Returns:
        Markdown-formatted string.
    """
    if not sources:
        return "No sources retrieved."

    lines = ["### Retrieved Sources\n"]
    for s in sources:
        lines.append(
            f"**Source {s['rank']}** "
            f"| 📁 {s['product_category']} "
            f"| 🏷️ {s['issue']} "
            f"| 🆔 Complaint #{s['complaint_id']} "
            f"| Score: `{s['score']:.4f}`\n"
        )
        lines.append(f"> {s['chunk_text'][:300]}{'...' if len(s['chunk_text']) > 300 else ''}\n")
        lines.append("---\n")

    return "\n".join(lines)


def clear_all() -> tuple:
    """Reset chat history, sources panel, and input box."""
    return [], "Sources will appear here after your first question.", ""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    title="CrediTrust Complaint Analyser",
    css="""
        #title-row { text-align: center; padding: 16px 0 8px 0; }
        #subtitle   { text-align: center; color: #555; margin-bottom: 12px; }
        #chatbox    { height: 480px; }
        #sources    { height: 480px; overflow-y: auto; }
        .source-box { font-size: 0.88rem; }
        footer      { display: none !important; }
    """
) as demo:

    gr.HTML("""
        <div id="title-row">
            <h1 style="color:#1F3864; font-size:2rem; font-weight:700;">
                🏦 CrediTrust Complaint Analyser
            </h1>
        </div>
        <p id="subtitle">
            Ask plain-English questions about customer complaints across
            Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.
        </p>
    """)

    gr.Markdown("---")

    with gr.Row():

        with gr.Column(scale=3):
            gr.Markdown("### 💬 Ask a Question")

            chatbot = gr.Chatbot(
                elem_id="chatbox",
                label="Conversation",
                bubble_full_width=False,
                show_label=False,
                avatar_images=(
                    None,                               
                    "https://img.icons8.com/color/48/bot.png"  
                ),
            )

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="e.g. Why are customers complaining about credit cards?",
                    label="Your Question",
                    scale=5,
                    lines=2,
                    show_label=False,
                )
                submit_btn = gr.Button(
                    "Ask ➤",
                    variant="primary",
                    scale=1,
                    min_width=80,
                )

            with gr.Row():
                product_filter = gr.Dropdown(
                    choices=PRODUCT_CHOICES,
                    value="All Products",
                    label="Filter by Product",
                    scale=2,
                )
                clear_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary",
                    scale=1,
                )

            gr.Markdown("**💡 Try one of these:**")
            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=question_box,
                label="",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📄 Retrieved Sources")
            sources_panel = gr.Markdown(
                value="Sources will appear here after your first question.",
                elem_id="sources",
                elem_classes=["source-box"],
            )

    gr.Markdown("---")

    gr.HTML("""
        <div style="text-align:center; color:#888; font-size:0.82rem; padding:8px 0;">
            CrediTrust Financial — Internal Use Only &nbsp;|&nbsp;
            Powered by FAISS + flan-t5-large + all-MiniLM-L6-v2 &nbsp;|&nbsp;
            Answers are grounded in retrieved complaint data only
        </div>
    """)

    history_state = gr.State([])

    submit_btn.click(
        fn=answer_question,
        inputs=[question_box, product_filter, history_state],
        outputs=[chatbot, sources_panel, question_box],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, product_filter, history_state],
        outputs=[chatbot, sources_panel, question_box],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[chatbot, sources_panel, question_box],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,        
        show_error=True,
    )