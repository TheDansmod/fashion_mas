r"""Inspect and move Qdrant collections: info, sample, snapshot, restore, copy-subset.

The collection defaults to the one for the configured embedding backend
(`settings.collection`: `fashion_gen` for siglip, `nova_fashion_gen` for nova);
`--collection` picks another. The URL defaults to `settings.vector_db.url`.

    uv run python scripts/qdrant_admin.py info
    uv run python scripts/qdrant_admin.py sample --count 3
    uv run python scripts/qdrant_admin.py snapshot --download data/qdrant_snapshots/
    uv run python scripts/qdrant_admin.py restore path/to/file.snapshot
    uv run python scripts/qdrant_admin.py copy-subset --from nova_fashion_gen \
        --to nova_small --ids-file ids.txt

Everything is local: nothing here touches AWS.
"""

import argparse
import random
import sys
from pathlib import Path

import httpx
from loguru import logger as log
from qdrant_client import QdrantClient, models

from _common import confirm, start

# snapshots of large collections take minutes; the default client timeout is seconds
TIMEOUT_S = 600


def parse_args(default_collection: str, default_url: str) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=default_url, help="Qdrant HTTP URL")
    parser.add_argument("--collection", default=default_collection)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="point count, vectors and status")
    sample = commands.add_parser(
        "sample", help="show random points (from the first 10,000)"
    )
    sample.add_argument("--count", type=int, default=1)
    snapshot = commands.add_parser("snapshot", help="create a collection snapshot")
    snapshot.add_argument(
        "--download", type=Path, metavar="DIR", help="also save it into DIR"
    )
    restore = commands.add_parser(
        "restore", help="restore the collection from a snapshot file"
    )
    restore.add_argument("snapshot_file", type=Path)
    copy = commands.add_parser(
        "copy-subset", help="copy some points to another collection (no re-embedding)"
    )
    copy.add_argument("--from", dest="source", required=True)
    copy.add_argument("--to", dest="target", required=True)
    copy.add_argument(
        "--ids-file",
        type=Path,
        required=True,
        help="one point id per line (an integer or a UUID)",
    )
    copy.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def info(client: QdrantClient, collection: str) -> None:
    """Print the collection's size, vectors and status."""
    details = client.get_collection(collection)
    vectors = details.config.params.vectors
    print(f"collection:      {collection}")
    print(f"points:          {client.count(collection, exact=True).count:,}")
    print(f"indexed vectors: {details.indexed_vectors_count:,}")
    print(f"segments:        {details.segments_count}")
    print(f"status:          {details.status.value}")
    for name, params in (vectors or {}).items():
        print(f"vector {name!r}: {params.size}-d, {params.distance.value}")


def sample(client: QdrantClient, collection: str, count: int) -> None:
    """Print random points: their payload and the start of each vector."""
    # point ids are integers in nova_fashion_gen but UUIDs in fashion_gen
    ids = [p.id for p in client.scroll(collection, limit=10_000, with_payload=False)[0]]
    for point in client.retrieve(
        collection, random.sample(ids, min(count, len(ids))), with_vectors=True
    ):
        print(f"\n--- point {point.id} ---")
        for name, vector in (point.vector or {}).items():
            print(f"{name} ({len(vector)}-d): {[round(v, 4) for v in vector[:6]]} ...")
        for key, value in (point.payload or {}).items():
            print(f"{key}: {str(value)[:120]}")


def snapshot(url: str, collection: str, download: Path | None) -> None:
    """Create a snapshot (stored by Qdrant under /qdrant/snapshots), optionally save it.

    With compose, /qdrant/snapshots is `QDRANT_SNAPSHOTS_DIR` (default
    `data/qdrant_snapshots/`), so the file is already on the host.
    """
    with httpx.Client(base_url=url, timeout=TIMEOUT_S) as http:
        response = http.post(f"/collections/{collection}/snapshots?wait=true")
        response.raise_for_status()
        name = response.json()["result"]["name"]
        log.info("Created snapshot {} of {}.", name, collection)
        if download is None:
            return
        download.mkdir(parents=True, exist_ok=True)
        path = download / name
        with (
            http.stream("GET", f"/collections/{collection}/snapshots/{name}") as stream,
            path.open("wb") as file,
        ):
            stream.raise_for_status()
            for chunk in stream.iter_bytes():
                file.write(chunk)
        log.info("Saved {}.", path)


def restore(url: str, client: QdrantClient, collection: str, file: Path) -> None:
    """Upload a snapshot file as `collection`, replacing it if it exists."""
    if client.collection_exists(collection) and not confirm(
        f"Collection '{collection}' exists and will be replaced by {file}."
    ):
        sys.exit("Not confirmed; nothing changed.")
    with httpx.Client(base_url=url, timeout=TIMEOUT_S) as http, file.open("rb") as f:
        response = http.post(
            f"/collections/{collection}/snapshots/upload",
            params={"priority": "snapshot", "wait": "true"},
            files={"snapshot": (file.name, f)},
        )
    response.raise_for_status()
    log.info("Restored {} from {}.", collection, file)
    info(client, collection)


def copy_subset(
    client: QdrantClient,
    source: str,
    target: str,
    ids_file: Path,
    batch_size: int,
) -> None:
    """Copy the listed points (vectors and payload) into `target`, creating it."""
    # integer ids (nova_fashion_gen uses catalogue indexes) or UUID strings
    ids = sorted(
        {int(x) if x.isdigit() else x for x in ids_file.read_text().split()}, key=str
    )
    if client.collection_exists(target):
        if not confirm(
            f"Collection '{target}' exists; points with the same ids are overwritten."
        ):
            sys.exit("Not confirmed; nothing changed.")
    else:
        params = client.get_collection(source).config.params
        client.create_collection(
            collection_name=target,
            vectors_config=params.vectors,
            on_disk_payload=params.on_disk_payload,
        )
        log.info("Created {} with the vector settings of {}.", target, source)
    copied = 0
    for first in range(0, len(ids), batch_size):
        points = client.retrieve(
            source, ids[first : first + batch_size], with_vectors=True
        )
        client.upsert(
            target,
            [
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )
        copied += len(points)
    log.info("Copied {} of {} requested points to {}.", copied, len(ids), target)
    if copied < len(ids):
        log.warning("{} ids were not in {}.", len(ids) - copied, source)


def main() -> None:
    """Run the chosen subcommand."""
    settings = start("qdrant-admin")
    args = parse_args(settings.collection, settings.vector_db.url)
    client = QdrantClient(url=args.url, prefer_grpc=False, timeout=TIMEOUT_S)
    match args.command:
        case "info":
            info(client, args.collection)
        case "sample":
            sample(client, args.collection, args.count)
        case "snapshot":
            snapshot(args.url, args.collection, args.download)
        case "restore":
            restore(args.url, client, args.collection, args.snapshot_file)
        case "copy-subset":
            copy_subset(
                client, args.source, args.target, args.ids_file, args.batch_size
            )


if __name__ == "__main__":
    main()
