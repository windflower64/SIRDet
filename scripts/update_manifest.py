"""Regenerate release hashes from Git's tracked/nonignored input file list."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
listing = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                         cwd=root, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
files = []
for name in sorted(set(filter(None, listing))):
    path = root / name
    if name == "MANIFEST.json" or not path.is_file():
        continue
    data = path.read_bytes()
    if path.suffix.lower() in {".py", ".md", ".sh", ".ps1", ".yaml", ".json", ".csv", ".txt", ".cpp", ".hpp", ".h"} or path.name in {"LICENSE", ".gitignore", ".gitattributes"}:
        data = data.replace(b"\r\n", b"\n")
    files.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
result = dict(package="SIRDet checkpoint-evaluation source bundle", generated_utc=datetime.now(timezone.utc).isoformat(),
              text_normalization="LF for source/documentation text; binaries unchanged", file_count=len(files), files=files)
(root / "MANIFEST.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"Wrote {len(files)} input hashes.")
