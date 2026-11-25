import os

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from huggingface_hub import snapshot_download


REPO_ID = "Qwen/Qwen3-Reranker-4B"
REPO_TYPE = "model"
REVISION = "main"
CACHE_DIR = "/home/fuxuanyi/.cache/huggingface/hub"
ENDPOINT = "https://huggingface.co"
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "1"))


def main() -> int:
    print(
        f"Downloading {REPO_ID}@{REVISION} to {CACHE_DIR} via "
        f"{os.environ['HTTPS_PROXY']} with {DOWNLOAD_WORKERS} workers",
        flush=True,
    )
    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=REVISION,
        cache_dir=CACHE_DIR,
        endpoint=ENDPOINT,
        max_workers=DOWNLOAD_WORKERS,
    )
    print(f"snapshot_path={snapshot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
