"""Chat-prompt builder for the LLaMA instruction model.

Applies the model's chat template with a generation prompt so the produced
string is exactly what the model saw at inference time.
"""

LLAMA_PROMPT_TEMPLATE = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\n"
    "Provide a concise solution with minimal explanation.\n\n"
    "- For complex problems (3 steps or more):\n"
    "Use this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem.\n\n"
    "Problem: {problem}\n"
)


def build_llama_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "user", "content": LLAMA_PROMPT_TEMPLATE.format(problem=question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

