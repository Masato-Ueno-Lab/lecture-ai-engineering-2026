"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask             # 会話履歴を保持する対話モード
    uv run python -m app.ask --chat      # 上と同じ対話モード

ツールは持たないが、対話モードでは前の質問と回答を会話履歴として保持する。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] 入力トークン: {usage.prompt_token_count or 0}, "
            f"出力トークン: {usage.candidates_token_count or 0}"
        )


def print_response_time(start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    print(f"[応答時間] {elapsed:.3f} 秒")


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    start_time = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    print_response_time(start_time)
    print_usage(response)
    return response.text


def chat() -> None:
    client = genai.Client(api_key=get_api_key())
    session = client.chats.create(model=MODEL)
    print("会話モードです。終了するには exit または quit と入力してください。")

    try:
        while True:
            try:
                question = input("[質問] ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            start_time = time.perf_counter()
            response = session.send_message(question)
            print_response_time(start_time)
            print_usage(response)
            print(f"[回答]\n{response.text}\n")
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保持して続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat or args.question is None:
            chat()
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
