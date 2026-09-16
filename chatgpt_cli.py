"""Ask a question in the terminal, send it to ChatGPT, print a summary of the call."""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-5.6-terra"

# Per-model OpenAI pricing, USD per 1M tokens: {model_id: (input_price, output_price)}.
# Source: https://developers.openai.com/api/docs/pricing - update here when OpenAI changes rates.
MODEL_PRICING_USD_PER_1M = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5": (1.25, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5-pro": (15.00, 120.00),
    "gpt-6-astra": (10.00, 50.00),
    "gpt-5.2-pro": (21.00, 168.00),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.5-pro": (30.00, 180.00),
}

USD_TO_INR = 95.75


def get_model_pricing(model_name):
    """Look up (input, output) USD-per-1M-token rates for a served model id.

    The API may return a dated/versioned id (e.g. 'gpt-4o-mini-2024-07-18'), so
    fall back to the longest known key that the served id starts with.
    """
    if model_name in MODEL_PRICING_USD_PER_1M:
        return MODEL_PRICING_USD_PER_1M[model_name]
    for key in sorted(MODEL_PRICING_USD_PER_1M, key=len, reverse=True):
        if model_name.startswith(key):
            return MODEL_PRICING_USD_PER_1M[key]
    return None

SYSTEM_PROMPT = (
    "You are a helpful assistant with a mildly sarcastic streak. You may open with one "
    "light, harmless quip, then drop the act and actually answer the user's query "
    "clearly, accurately and completely. Never let the sarcasm get in the way of the " 
    "answer, and never be mean about it."
)

USER_PROMPT = (
    "Below is my query. Yes, I could have searched for it myself, but here we are. "
    "Please answer it properly.\n\nQuery: "
)


def main():
    if not API_KEY:
        print("Error: OPENAI_API_KEY is not set in the environment.", file=sys.stderr)
        return 1

    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("No query entered. Nothing to ask.")
        return 1

    client = OpenAI(api_key=API_KEY)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT + user_query},
            ],
        )
    except OpenAIError as exc:
        print("OpenAI API call failed: {}".format(exc), file=sys.stderr)
        return 1

    answer = response.choices[0].message.content
    usage = response.usage

    pricing = get_model_pricing(response.model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nUser query:\n  {}".format(user_query))
    print("\nSystem prompt:\n  {}".format(SYSTEM_PROMPT))
    print("\nUser prompt (static part):\n  {}".format(USER_PROMPT.strip()))
    print("\nModel: {}".format(response.model))
    print("\nToken usage:")
    print("  Input  tokens: {}".format(usage.prompt_tokens))
    print("  Output tokens: {}".format(usage.completion_tokens))
    print("  Total  tokens: {}".format(usage.total_tokens))
    if pricing is None:
        print("\nEstimated cost: unknown (no pricing entry for model '{}')".format(response.model))
    else:
        input_rate, output_rate = pricing
        input_cost_usd = (usage.prompt_tokens / 1_000_000) * input_rate
        output_cost_usd = (usage.completion_tokens / 1_000_000) * output_rate
        total_cost_usd = input_cost_usd + output_cost_usd
        total_cost_inr = total_cost_usd * USD_TO_INR

        print("\nRate for {} (USD per 1M tokens): ${:.2f} in / ${:.2f} out".format(
            response.model, input_rate, output_rate))
        print("Estimated cost:")
        print("  USD: ${:.6f}".format(total_cost_usd))
        print("  INR: Rs. {:.4f}  (at Rs. {}/USD)".format(total_cost_inr, USD_TO_INR))
    print("\n" + "-" * 70)
    print("LLM response:")
    print("-" * 70)
    print(answer)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
