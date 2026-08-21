from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "llm_benchmark.api:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    main()