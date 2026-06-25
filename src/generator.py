"""
generator.py
------------
Updated to load flan-t5 directly via AutoModelForSeq2SeqLM instead of
the transformers pipeline, which dropped text2text-generation support
in newer versions.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer questions about customer complaints.

Use ONLY the complaint excerpts provided below to formulate your answer.
Do not use any outside knowledge. If the context does not contain enough
information to answer the question, say: "I don't have enough information
in the retrieved complaints to answer this question confidently."

Where possible, reference which product or issue category the evidence comes from.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, context: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")
    return PROMPT_TEMPLATE.format(context=context, question=question.strip())


def load_generator(
    model_name: str = "google/flan-t5-large",
    max_new_tokens: int = 300,
    device: int = -1,
) -> dict:
    """
    Load flan-t5 directly via AutoTokenizer + AutoModelForSeq2SeqLM.

    Design choice: Direct model loading is used instead of the
    transformers pipeline() because newer transformers versions
    removed text2text-generation from the pipeline registry.
    Returning a dict keeps the interface clean for generate_answer().

    Args:
        model_name:     HuggingFace model identifier.
        max_new_tokens: Maximum tokens in the generated answer.
        device:         -1 = CPU, 0 = first GPU.

    Returns:
        Dict with keys: tokenizer, model, max_new_tokens, device.
    """
    try:
        print(f"Loading tokenizer and model: {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        torch_device = torch.device("cuda" if device >= 0 and torch.cuda.is_available() else "cpu")
        model = model.to(torch_device)
        model.eval()

        print(f"Generator loaded: {model_name} on {torch_device}")
        return {
            "tokenizer":      tokenizer,
            "model":          model,
            "max_new_tokens": max_new_tokens,
            "device":         torch_device,
        }
    except Exception as e:
        raise OSError(
            f"Failed to load generator model '{model_name}'.\n"
            f"Check your internet connection or model name.\n"
            f"Original error: {e}"
        )


def generate_answer(prompt: str, generator: dict) -> str:
    """
    Generate an answer from the filled prompt using flan-t5.

    Args:
        prompt:    Complete prompt string from build_prompt().
        generator: Dict returned by load_generator().

    Returns:
        Generated answer string.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    try:
        tokenizer      = generator["tokenizer"]
        model          = generator["model"]
        max_new_tokens = generator["max_new_tokens"]
        device         = generator["device"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024       # flan-t5-large max input length
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if not answer:
            return "The model returned an empty response. Try rephrasing your question."

        return answer

    except Exception as e:
        raise RuntimeError(f"Generation failed.\nOriginal error: {e}")