"""Download exact PyPI wheels without inherited proxies or the system DNS.

This helper is intended for GPU nodes where a transparent proxy returns fake
DNS addresses.  curl resolves every HTTPS host through a DoH endpoint whose
bootstrap address is fixed with ``--resolve``; TLS SNI and certificate checks
still use the original hostname.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

PROXY_VARIABLES = (
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+", help="exact specs such as sympy==1.14.0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interface", default="eno2")
    parser.add_argument("--doh-host", default="dns.alidns.com")
    parser.add_argument("--doh-ip", default="223.5.5.5")
    return parser.parse_args()


def _split_exact_spec(spec: str) -> tuple[str, str]:
    name, separator, version = spec.partition("==")
    if not separator or not name or not version or "==" in version:
        raise ValueError(f"package must use one exact name==version spec: {spec!r}")
    return name, version


def _network_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in PROXY_VARIABLES:
        environment.pop(variable, None)
    return environment


def _curl_base(args: argparse.Namespace) -> list[str]:
    return [
        "curl",
        "-q",
        "--proxy",
        "",
        "--noproxy",
        "*",
        "-4",
        "--interface",
        args.interface,
        "--doh-url",
        f"https://{args.doh_host}/dns-query",
        "--resolve",
        f"{args.doh_host}:443:{args.doh_ip}",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--fail",
        "--show-error",
        "--location",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
    ]


def _metadata(spec: str, args: argparse.Namespace) -> dict[str, object]:
    name, version = _split_exact_spec(spec)
    command = [
        *_curl_base(args),
        "--silent",
        f"https://pypi.org/pypi/{name}/{version}/json",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        env=_network_environment(),
    )
    return json.loads(completed.stdout)


def _select_wheel(metadata: dict[str, object]) -> dict[str, object]:
    supported = list(sys_tags())
    priority = {tag: index for index, tag in enumerate(supported)}
    candidates: list[tuple[int, dict[str, object]]] = []
    for entry in metadata["urls"]:  # type: ignore[index]
        if entry["packagetype"] != "bdist_wheel":
            continue
        _, _, _, tags = parse_wheel_filename(entry["filename"])
        matching = [priority[tag] for tag in tags if tag in priority]
        if matching:
            candidates.append((min(matching), entry))
    if not candidates:
        name = metadata["info"]["name"]  # type: ignore[index]
        version = metadata["info"]["version"]  # type: ignore[index]
        raise RuntimeError(f"no compatible wheel found for {name}=={version}")
    candidates.sort(key=lambda item: (item[0], item[1]["filename"]))
    return candidates[0][1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(entry: dict[str, object], args: argparse.Namespace) -> Path:
    output = args.output_dir / str(entry["filename"])
    expected = str(entry["digests"]["sha256"])  # type: ignore[index]
    if output.is_file() and _sha256(output) == expected:
        print(f"cached {output.name} sha256={expected}")
        return output

    command = [
        *_curl_base(args),
        "--continue-at",
        "-",
        "--output",
        str(output),
        str(entry["url"]),
    ]
    subprocess.run(command, check=True, env=_network_environment())
    actual = _sha256(output)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {output.name}: expected {expected}, got {actual}"
        )
    print(f"downloaded {output.name} sha256={actual}")
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.packages:
        entry = _select_wheel(_metadata(spec, args))
        _download(entry, args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"error: {exc}") from exc
