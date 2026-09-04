"""Ask a question in the terminal, send it to ChatGPT, print a summary of the call."""

import sys

from openai import OpenAI, OpenAIError

# Hard-coded as requested. Replace with your real key before running.
API_KEY = "sk-REPLACE_ME_WITH_YOUR_KEY"
MODEL = "gpt-4o-mini"

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
