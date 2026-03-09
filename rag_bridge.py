#!/var/home/apollo/dev/rags_to_riches/.venv/bin/python3
"""
RAG-Anything bridge: reads a JSON op from stdin, runs it, writes JSON result to stdout.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=False)  # env vars from compose take priority


def get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


async def run_ingest(paths: list[str], recursive: bool) -> dict:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed

    working_dir = get_env("RAG_WORKING_DIR", str(Path.home() / ".rag_storage"))
    output_dir = get_env("RAG_OUTPUT_DIR", str(Path(working_dir) / "output"))
    api_base = get_env("LLM_API_BASE", "https://api.openai.com/v1")
    llm_model = get_env("LLM_MODEL", "gpt-4o-mini")
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = get_env("OPENAI_API_KEY", "")

    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=working_dir,
        output_dir=output_dir,
    )

    async def llm_func(prompt, system_prompt=None, **kwargs):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=api_base,
            **kwargs,
        )

    async def embed_func(texts: list[str]) -> list[list[float]]:
        return await openai_embed(
            texts,
            model=embedding_model,
            api_key=api_key,
            base_url=api_base,
        )

    rag = RAGAnything(config=config, llm_model_func=llm_func, embedding_func=embed_func)

    added = []
    errors = []

    for p in paths:
        path = Path(p)
        try:
            if path.is_dir():
                await rag.process_folder_complete(
                    folder_path=str(path),
                    recursive=recursive,
                    output_dir=output_dir,
                )
                added.append(str(path))
            elif path.is_file():
                await rag.process_document_complete(
                    file_path=str(path),
                    output_dir=output_dir,
                )
                added.append(str(path))
            else:
                errors.append(f"Path not found: {p}")
        except Exception as e:
            errors.append(f"{p}: {e}")

    return {"added": added, "errors": errors}


async def run_query(query: str, mode: str) -> dict:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed

    working_dir = get_env("RAG_WORKING_DIR", str(Path.home() / ".rag_storage"))
    output_dir = get_env("RAG_OUTPUT_DIR", str(Path(working_dir) / "output"))
    api_base = get_env("LLM_API_BASE", "https://api.openai.com/v1")
    llm_model = get_env("LLM_MODEL", "gpt-4o-mini")
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = get_env("OPENAI_API_KEY", "")

    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=working_dir,
        output_dir=output_dir,
    )

    async def llm_func(prompt, system_prompt=None, **kwargs):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=api_base,
            **kwargs,
        )

    async def embed_func(texts: list[str]) -> list[list[float]]:
        return await openai_embed(
            texts,
            model=embedding_model,
            api_key=api_key,
            base_url=api_base,
        )

    rag = RAGAnything(config=config, llm_model_func=llm_func, embedding_func=embed_func)

    result = await rag.query(query, mode=mode)
    return {"result": result}


def main():
    line = sys.stdin.readline().strip()
    if not line:
        print(json.dumps({"error": "No input received"}))
        sys.exit(1)

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(1)

    op = payload.get("op")

    try:
        if op == "ingest":
            paths = payload.get("paths", [])
            recursive = payload.get("recursive", True)
            result = asyncio.run(run_ingest(paths, recursive))
        elif op == "query":
            query = payload.get("query", "")
            mode = payload.get("mode", "mix")
            result = asyncio.run(run_query(query, mode))
        else:
            print(json.dumps({"error": f"Unknown op: {op}"}))
            sys.exit(1)

        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
