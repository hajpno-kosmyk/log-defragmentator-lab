"""Log Defragmentator Lab — log_fix.py

Transform chaotic Facebook/Messenger JSON exports into clean, searchable archives.

Features
--------
- Fix Facebook's UTF-8 mojibake (Å¼ → ż, Ä… → ą, etc.)
- Export to Markdown, CSV, JSON, or styled HTML
- Full-text search across all conversations
- Statistics dashboard with activity analytics
- Progress bars, parallel processing, streaming JSON
- Hash-based caching for fast re-runs
- Custom Jinja2 HTML templates

Outputs
-------
- One Markdown file per thread (chronological)
- An index Markdown with links + stats
- Optional: CSV, JSON, HTML exports
- Optional: Statistics dashboard

Usage
-----
  python log_fix.py --input ./inbox --out ./archive
  python log_fix.py --input ./inbox --out ./archive --html --csv --stats
  python log_fix.py --input ./inbox --search "keyword"

See docs/EXAMPLES.md for detailed usage examples.

Requirements
------------
- Python 3.10+
- Optional: tqdm (progress bars), jinja2 (templates), ijson (streaming), pyyaml (config)

License: MIT
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import json
import logging
import os
import pickle
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from functools import partial, lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Optional YAML support for config files
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Optional Jinja2 support for HTML templates
try:
    from jinja2 import Template, Environment, BaseLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Optional ijson for streaming JSON parsing
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False


# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# =============================================================================
# COMPILED REGEX PATTERNS
# =============================================================================

class Patterns:
    """Pre-compiled regex patterns for text processing.

    Centralizes all regex patterns for:
    - Consistent behavior across the codebase
    - Single compile operation (performance)
    - Easy testing and modification
    """
    # Text cleaning patterns
    CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    WHITESPACE = re.compile(r"[\t\r\f\v ]+")

    # Filename sanitization (Windows-safe)
    WINDOWS_UNSAFE = re.compile(r'[\\/:*?"<>|]')
    MULTI_UNDERSCORE = re.compile(r"_+")

    # JSON auto-fix patterns
    MISSING_COMMA = re.compile(r'(\d+)\n(\s*"sender_name")')
    TRUNCATED_MSG = re.compile(
        r'\{\s*"sender_name":\s*"[^"]*",\s*"timestamp_ms":\s*\d+\s*"sender_name"',
        re.DOTALL
    )
    TRAILING_COMMA = re.compile(r',(\s*[}\]])')
    UNESCAPED_NEWLINE = re.compile(r'("content":\s*"[^"]*)\n([^"]*")')


# Legacy aliases for backward compatibility
CONTROL_CHARS_RE = Patterns.CONTROL_CHARS
WHITESPACE_RE = Patterns.WHITESPACE


def fix_facebook_encoding(s: str) -> str:
    """Fix Facebook's double-encoded UTF-8 mojibake.

    Facebook Messenger exports often encode UTF-8 text as latin-1, causing
    Polish characters like 'ż' to appear as 'Å¼', 'ą' as 'Ä…', etc.

    This function detects and fixes this by re-encoding latin-1 bytes as UTF-8.
    If the text is already correct or conversion fails, returns original string.

    Common mojibake patterns this fixes:
        Å¼ -> ż,  Ä… -> ą,  Ä™ -> ę,  Å› -> ś,  Ã³ -> ó
        Åº -> ź,  Å„ -> ń,  Å‚ -> ł,  Ä‡ -> ć
    """
    if not s:
        return s

    # Quick heuristic: look for common mojibake byte sequences
    # These appear when UTF-8 bytes are misinterpreted as latin-1
    mojibake_indicators = (
        'Ä', 'Å', 'Ã',  # Common prefixes in double-encoded Polish/European text
        'Ä…', 'Ä™', 'Å¼', 'Å›', 'Ã³', 'Åº', 'Å„', 'Å‚', 'Ä‡',  # Specific Polish
        'Ã¼', 'Ã¶', 'Ã¤', 'ÃŸ',  # German
        'Ã©', 'Ã¨', 'Ã ',  # French
    )

    # Only attempt fix if mojibake indicators are present
    if not any(indicator in s for indicator in mojibake_indicators):
        return s

    try:
        # Encode as latin-1 (ISO-8859-1) to get raw bytes, then decode as UTF-8
        fixed = s.encode('latin-1').decode('utf-8')
        return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Conversion failed - text might be partially corrupted or already correct
        return s


@dataclass
class CleanMessage:
    timestamp_ms: int
    dt_iso: str
    sender: str
    content: str
    reactions: str
    attachments: str
    attachment_details: list[dict] = field(default_factory=list)  # Detailed attachment info


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def clean_text(s: str) -> str:
    """Normalize Messenger text for archiving.

    - Fix Facebook's double-encoded UTF-8 (mojibake)
    - Remove ASCII control chars
    - Unescape HTML entities (some exports contain them)
    - Normalize whitespace (keep newlines)
    - Trim trailing spaces per line

    Polish diacritics are preserved (UTF-8)."""

    s = _safe_str(s)
    s = fix_facebook_encoding(s)  # Fix mojibake before other processing
    s = html.unescape(s)
    s = CONTROL_CHARS_RE.sub("", s)

    # Normalize line endings and strip trailing whitespace per line
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", ln).rstrip() for ln in s.split("\n")]

    # Drop excessive blank lines (3+ -> 2)
    out_lines = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                out_lines.append("")
        else:
            blank_run = 0
            out_lines.append(ln)

    return "\n".join(out_lines).strip("\n")


def ts_to_iso_local(ts_ms: int, tz: Optional[timezone] = None) -> str:
    """Convert timestamp_ms to ISO string.

    If tz is None, use local timezone via datetime.fromtimestamp (OS local).
    """
    seconds = ts_ms / 1000.0
    if tz is None:
        dt = datetime.fromtimestamp(seconds)
    else:
        dt = datetime.fromtimestamp(seconds, tz=tz)
    return dt.isoformat(timespec="seconds")


def summarize_reactions(msg: dict) -> str:
    reactions = msg.get("reactions") or []
    if not reactions:
        return ""
    # Common fields: reaction, actor
    parts = []
    for r in reactions:
        rx = clean_text(_safe_str(r.get("reaction")))
        actor = clean_text(_safe_str(r.get("actor")))
        if rx and actor:
            parts.append(f"{rx} by {actor}")
        elif rx:
            parts.append(rx)
    return "; ".join(parts)


def summarize_attachments(msg: dict) -> tuple[str, list[dict]]:
    """Summarize non-text payloads and extract detailed attachment info.

    Returns:
        tuple: (summary_string, list_of_attachment_details)
    """
    bits = []
    details = []

    def extract_filename(uri: str) -> str:
        """Extract filename from URI path."""
        if not uri:
            return ""
        return uri.split("/")[-1].split("?")[0]

    # Media often uses lists: photos/videos/audio_files/files/gifs/stickers
    photos = msg.get("photos") or []
    videos = msg.get("videos") or []
    audio = msg.get("audio_files") or []
    files = msg.get("files") or []
    gifs = msg.get("gifs") or []
    stickers = msg.get("sticker")

    if photos:
        filenames = [extract_filename(p.get("uri", "")) for p in photos]
        bits.append(f"{len(photos)} photo(s): {', '.join(f for f in filenames if f)}" if any(filenames) else f"{len(photos)} photo(s)")
        for p in photos:
            details.append({"type": "photo", "uri": p.get("uri", ""), "filename": extract_filename(p.get("uri", ""))})

    if videos:
        filenames = [extract_filename(v.get("uri", "")) for v in videos]
        bits.append(f"{len(videos)} video(s): {', '.join(f for f in filenames if f)}" if any(filenames) else f"{len(videos)} video(s)")
        for v in videos:
            details.append({"type": "video", "uri": v.get("uri", ""), "filename": extract_filename(v.get("uri", "")), "thumbnail": v.get("thumbnail", {}).get("uri", "")})

    if audio:
        filenames = [extract_filename(a.get("uri", "")) for a in audio]
        bits.append(f"{len(audio)} audio: {', '.join(f for f in filenames if f)}" if any(filenames) else f"{len(audio)} audio file(s)")
        for a in audio:
            details.append({"type": "audio", "uri": a.get("uri", ""), "filename": extract_filename(a.get("uri", ""))})

    if files:
        filenames = [extract_filename(f.get("uri", "")) for f in files]
        bits.append(f"{len(files)} file(s): {', '.join(f for f in filenames if f)}" if any(filenames) else f"{len(files)} file(s)")
        for f in files:
            details.append({"type": "file", "uri": f.get("uri", ""), "filename": extract_filename(f.get("uri", ""))})

    if gifs:
        filenames = [extract_filename(g.get("uri", "")) for g in gifs]
        bits.append(f"{len(gifs)} gif(s): {', '.join(f for f in filenames if f)}" if any(filenames) else f"{len(gifs)} gif(s)")
        for g in gifs:
            details.append({"type": "gif", "uri": g.get("uri", "")})

    if stickers:
        sticker_uri = stickers.get("uri", "") if isinstance(stickers, dict) else ""
        bits.append("sticker")
        details.append({"type": "sticker", "uri": sticker_uri})

    # Shares/links
    share = msg.get("share")
    if isinstance(share, dict):
        link = share.get("link") or ""
        if link:
            bits.append(f"share: {link}")
            details.append({"type": "share", "link": link, "text": share.get("share_text", "")})
        else:
            bits.append("share")
            details.append({"type": "share", "text": share.get("share_text", "")})

    return "; ".join(bits), details


def parse_message(msg: dict) -> Optional[CleanMessage]:
    if not isinstance(msg, dict):
        return None

    # Handle both standard format (timestamp_ms) and alternate format (timestamp)
    ts_ms = msg.get("timestamp_ms")
    if ts_ms is None:
        ts_ms = msg.get("timestamp")  # Alternate format uses "timestamp"

    if ts_ms is None:
        return None

    try:
        # Handle string timestamps or floats
        ts_ms_int = int(float(ts_ms))
    except (ValueError, TypeError):
        return None

    # Handle both standard format (sender_name) and alternate format (senderName)
    sender = clean_text(_safe_str(msg.get("sender_name") or msg.get("senderName")))

    # Handle both standard format (content) and alternate format (text)
    content_raw = msg.get("content")
    if content_raw is None:
        content_raw = msg.get("text")  # Alternate format uses "text"

    # Some messages have no content (only attachments)
    content = clean_text(_safe_str(content_raw))
    reactions = summarize_reactions(msg)
    attachments, attachment_details = summarize_attachments(msg)

    dt_iso = ts_to_iso_local(ts_ms_int)

    # If truly empty, skip unless attachments exist
    if content.strip() == "" and attachments.strip() == "" and reactions.strip() == "":
        return None

    return CleanMessage(
        timestamp_ms=ts_ms_int,
        dt_iso=dt_iso,
        sender=sender or "(unknown)",
        content=content,
        reactions=reactions,
        attachments=attachments,
        attachment_details=attachment_details,
    )


def iter_message_json_files(thread_dir: Path) -> list[Path]:
    files = sorted(thread_dir.glob("message_*.json"))
    # Some exports might name it differently; include any .json if needed.
    if not files:
        files = sorted(p for p in thread_dir.glob("*.json") if p.is_file())
    return files


# =============================================================================
# STREAMING JSON PARSER (for large files)
# =============================================================================

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def stream_json_messages(file_path: Path) -> Iterable[dict]:
    """Stream messages from a JSON file without loading entire file into memory.

    Uses ijson for incremental parsing when available.
    Falls back to standard json.load() for smaller files or when ijson unavailable.

    Args:
        file_path: Path to the JSON file

    Yields:
        Individual message dictionaries
    """
    if not IJSON_AVAILABLE:
        # Fallback: load entire file
        try:
            data, _ = load_json_with_fixes(file_path)
            if isinstance(data, dict):
                msgs = data.get("messages")
                if isinstance(msgs, list):
                    yield from msgs
        except Exception:
            pass
        return

    # Use ijson for streaming
    try:
        with open(file_path, 'rb') as f:
            # Stream messages array items one at a time
            for msg in ijson.items(f, 'messages.item'):
                yield msg
    except Exception as e:
        logger.debug(f"Streaming parse failed for {file_path}: {e}, falling back to standard load")
        # Fallback on error
        try:
            data, _ = load_json_with_fixes(file_path)
            if isinstance(data, dict):
                msgs = data.get("messages")
                if isinstance(msgs, list):
                    yield from msgs
        except Exception:
            pass


def load_thread_streaming(thread_dir: Path, size_threshold_mb: float = 50.0) -> tuple[str, list[str], list[CleanMessage]]:
    """Load a thread using streaming for large files.

    Uses streaming JSON parsing for files larger than size_threshold_mb.
    This dramatically reduces memory usage for very large conversation exports.

    Args:
        thread_dir: Path to thread directory
        size_threshold_mb: Files larger than this use streaming (default: 50MB)

    Returns:
        Tuple of (title, participants, messages)
    """
    json_files = iter_message_json_files(thread_dir)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {thread_dir}")

    all_messages: list[CleanMessage] = []
    title = thread_dir.name
    participants_set = set()

    for fp in json_files:
        file_size = get_file_size_mb(fp)
        use_streaming = IJSON_AVAILABLE and file_size > size_threshold_mb

        if use_streaming:
            logger.debug(f"Using streaming parser for {fp.name} ({file_size:.1f}MB)")

            # For streaming, we need to extract metadata separately
            # Quick peek at the start for title/participants
            try:
                with open(fp, 'rb') as f:
                    parser = ijson.parse(f)
                    for prefix, event, value in parser:
                        if prefix == 'title' and event == 'string':
                            title = clean_text(value) or title
                        elif prefix == 'participants.item.name' and event == 'string':
                            name = clean_text(value)
                            if name:
                                participants_set.add(name)
                        elif prefix == 'participants.item' and event == 'string':
                            # Handle string participant format
                            name = clean_text(value)
                            if name:
                                participants_set.add(name)
                        # Stop once we hit messages
                        elif prefix == 'messages':
                            break
            except Exception as e:
                logger.debug(f"Streaming metadata parse failed for {fp.name}: {e}")

            # Now stream messages
            for msg in stream_json_messages(fp):
                cm = parse_message(msg)
                if cm:
                    all_messages.append(cm)
        else:
            # Standard loading for smaller files
            try:
                data, fixes = load_json_with_fixes(fp)
                if fixes:
                    logger.debug(f"Auto-fixed {fp.name}: {', '.join(fixes)}")
            except Exception as e:
                logger.debug(f"Skipping malformed file {fp.name}: {e}")
                continue

            if isinstance(data, dict):
                title = clean_text(_safe_str(data.get("title"))) or title

                participants = data.get("participants")
                if participants and isinstance(participants, list):
                    for p in participants:
                        if isinstance(p, str):
                            name = clean_text(p)
                        elif isinstance(p, dict):
                            name = clean_text(_safe_str(p.get("name")))
                        else:
                            name = ""
                        if name:
                            participants_set.add(name)

                messages = data.get("messages")
                if messages and isinstance(messages, list):
                    for m in messages:
                        cm = parse_message(m)
                        if cm:
                            all_messages.append(cm)

    all_messages.sort(key=lambda x: x.timestamp_ms)
    participants_list = sorted(participants_set)

    if not title.strip() and participants_list:
        title = ", ".join(participants_list[:4]) + ("…" if len(participants_list) > 4 else "")

    return title, participants_list, all_messages


def load_thread(thread_dir: Path) -> tuple[str, list[str], list[CleanMessage]]:
    """Load one thread folder and return (thread_title, participants, messages)."""

    json_files = iter_message_json_files(thread_dir)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {thread_dir}")

    all_messages: list[CleanMessage] = []
    title = thread_dir.name
    participants_set = set()

    for fp in json_files:
        try:
            data, fixes = load_json_with_fixes(fp)
            if fixes:
                logger.debug(f"Auto-fixed {fp.name}: {', '.join(fixes)}")
        except Exception as e:
            logger.debug(f"Skipping malformed file {fp.name}: {e}")
            continue

        if isinstance(data, dict):
            title = clean_text(_safe_str(data.get("title"))) or title

            participants = data.get("participants")
            if participants and isinstance(participants, list):
                for p in participants:
                    # Handle both formats: string or dict with "name" key
                    if isinstance(p, str):
                        name = clean_text(p)
                    elif isinstance(p, dict):
                        name = clean_text(_safe_str(p.get("name")))
                    else:
                        name = ""
                    if name:
                        participants_set.add(name)

            messages = data.get("messages")
            if messages and isinstance(messages, list):
                for m in messages:
                    cm = parse_message(m)
                    if cm:
                        all_messages.append(cm)
        else:
            # Unexpected structure
            logger.debug(f"Skipping {fp.name}: not a dictionary")
            continue

    # Messenger exports usually list messages newest->oldest; we want oldest->newest
    all_messages.sort(key=lambda x: x.timestamp_ms)
    participants_list = sorted(participants_set)

    # If title is empty, fallback to participants
    if not title.strip() and participants_list:
        title = ", ".join(participants_list[:4]) + ("…" if len(participants_list) > 4 else "")

    return title, participants_list, all_messages


def filter_messages_by_date(
    msgs: list[CleanMessage],
    after: Optional[datetime] = None,
    before: Optional[datetime] = None
) -> list[CleanMessage]:
    """Filter messages by date range.

    Args:
        msgs: List of messages to filter
        after: Only include messages after this datetime (inclusive)
        before: Only include messages before this datetime (inclusive)

    Returns:
        Filtered list of messages
    """
    if not after and not before:
        return msgs

    filtered = []
    for m in msgs:
        msg_dt = datetime.fromtimestamp(m.timestamp_ms / 1000.0)

        if after and msg_dt < after:
            continue
        if before and msg_dt > before:
            continue

        filtered.append(m)

    return filtered


def parse_date_arg(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string in YYYY-MM-DD format."""
    if not date_str:
        return None

    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug.

    - Lowercases the string
    - Replaces Windows-unsafe characters with hyphens
    - Replaces whitespace with underscores
    - Collapses multiple underscores
    - Truncates to 140 characters
    """
    name = name.strip().lower()
    # Keep Polish letters, replace separators; Windows-safe
    name = Patterns.WINDOWS_UNSAFE.sub("-", name)
    name = Patterns.WHITESPACE.sub("_", name)
    name = Patterns.MULTI_UNDERSCORE.sub("_", name)
    return name[:140] if len(name) > 140 else name


def write_thread_markdown(out_dir: Path, thread_id: str, title: str, participants: list[str], msgs: list[CleanMessage]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{thread_id}.md"

    header = [
        f"# {title}",
        "",
        f"**Thread ID:** `{thread_id}`",
        f"**Participants:** {', '.join(participants) if participants else '(unknown)' }",
        f"**Messages:** {len(msgs)}",
        "",
        "---",
        "",
    ]

    lines = header

    for m in msgs:
        lines.append(f"## {m.dt_iso} — {m.sender}")
        if m.reactions:
            lines.append(f"- Reactions: {m.reactions}")
        if m.attachments:
            lines.append(f"- Attachments: {m.attachments}")
        lines.append("")
        lines.append(m.content if m.content else "*(no text)*")
        lines.append("")
        lines.append("---")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def write_thread_csv(out_dir: Path, thread_id: str, msgs: list[CleanMessage]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{thread_id}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dt_iso", "sender", "content", "reactions", "attachments", "timestamp_ms"])
        for m in msgs:
            w.writerow([m.dt_iso, m.sender, m.content, m.reactions, m.attachments, m.timestamp_ms])
    return csv_path


def write_thread_json(out_dir: Path, thread_id: str, title: str, participants: list[str], msgs: list[CleanMessage]) -> Path:
    """Export thread data as JSON for programmatic processing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{thread_id}.json"

    data = {
        "thread_id": thread_id,
        "title": title,
        "participants": participants,
        "message_count": len(msgs),
        "date_range": {
            "first": msgs[0].dt_iso if msgs else None,
            "last": msgs[-1].dt_iso if msgs else None,
        },
        "messages": [
            {
                "timestamp_ms": m.timestamp_ms,
                "dt_iso": m.dt_iso,
                "sender": m.sender,
                "content": m.content,
                "reactions": m.reactions,
                "attachments": m.attachments,
                "attachment_details": m.attachment_details,
            }
            for m in msgs
        ],
    }

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


@dataclass
class ThreadStats:
    """Statistics for a processed thread."""
    thread_id: str
    title: str
    message_count: int
    md_relpath: str
    first_date: str = ""
    last_date: str = ""
    participants: list[str] = field(default_factory=list)
    messages_per_sender: dict[str, int] = field(default_factory=dict)


def compute_thread_stats(
    thread_id: str,
    title: str,
    participants: list[str],
    msgs: list[CleanMessage],
    md_relpath: str
) -> ThreadStats:
    """Compute statistics for a thread."""
    sender_counts: Counter[str] = Counter()
    for m in msgs:
        sender_counts[m.sender] += 1

    return ThreadStats(
        thread_id=thread_id,
        title=title,
        message_count=len(msgs),
        md_relpath=md_relpath,
        first_date=msgs[0].dt_iso[:10] if msgs else "",
        last_date=msgs[-1].dt_iso[:10] if msgs else "",
        participants=participants,
        messages_per_sender=dict(sender_counts),
    )


def write_index(out_dir: Path, items: list[ThreadStats]) -> Path:
    """Write index with enhanced statistics."""
    idx = out_dir / "INDEX.md"

    total_messages = sum(item.message_count for item in items)
    total_threads = len(items)

    lines = [
        "# Messenger Archive Index",
        "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"**Total Threads:** {total_threads}",
        f"**Total Messages:** {total_messages:,}",
        "",
        "---",
        "",
        "## Threads by Message Count",
        "",
        "| Messages | Thread | Date Range | Top Sender | File |",
        "|---:|---|---|---|---|",
    ]

    for item in sorted(items, key=lambda x: x.message_count, reverse=True):
        safe_title = item.title.replace("|", "\\|")
        date_range = f"{item.first_date} to {item.last_date}" if item.first_date else "N/A"

        # Find top sender
        top_sender = ""
        if item.messages_per_sender:
            top = max(item.messages_per_sender.items(), key=lambda x: x[1])
            top_sender = f"{top[0]} ({top[1]})"

        lines.append(f"| {item.message_count:,} | {safe_title} | {date_range} | {top_sender} | [{item.thread_id}]({item.md_relpath}) |")

    idx.write_text("\n".join(lines), encoding="utf-8")
    return idx


def find_thread_dirs(root: Path) -> list[Path]:
    """Heuristic: a thread dir is any dir containing message JSON files.

    Detects both standard (message_*.json) and non-standard naming patterns
    like laura_messenger_logs_*.json used in some e2ee exports.
    """
    thread_dirs = []
    for d in root.rglob("*"):
        if d.is_dir():
            # Check for standard message_*.json first
            if any(d.glob("message_*.json")):
                thread_dirs.append(d)
            # Also check for other messenger log patterns (e2ee exports)
            elif any(d.glob("*messenger_logs*.json")) or any(d.glob("*_messages*.json")):
                thread_dirs.append(d)

    # If none found via rglob, check if root itself is a thread folder
    if not thread_dirs:
        if any(root.glob("message_*.json")) or any(root.glob("*messenger_logs*.json")):
            thread_dirs = [root]

    return sorted(set(thread_dirs))


def find_standalone_json_files(root: Path) -> list[Path]:
    """Find standalone JSON conversation files (not in message_*.json format).

    Some Facebook exports have individual JSON files per conversation
    directly in a folder, not in subdirectories.
    """
    json_files = []

    # Look for JSON files directly in the root (not in subdirectories)
    for f in root.glob("*.json"):
        if f.is_file() and not f.name.startswith("message_"):
            # Check if it looks like a conversation file
            try:
                # Quick check for conversation structure
                content = f.read_text(encoding="utf-8", errors="ignore")[:500]
                if '"participants"' in content or '"messages"' in content:
                    json_files.append(f)
            except:
                pass

    return sorted(json_files)


def load_standalone_json(json_file: Path) -> tuple[str, list[str], list[CleanMessage]]:
    """Load a standalone JSON conversation file.

    Handles two participant formats:
    - Standard: [{"name": "Alice"}, {"name": "Bob"}]
    - Alternate: ["Alice", "Bob"]

    Also handles alternate field name "threadName" for title.
    """
    try:
        data, fixes = load_json_with_fixes(json_file)
    except Exception as e:
        raise ValueError(f"Failed to parse {json_file}: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid format in {json_file}")

    # Extract title from filename or data (try both "title" and "threadName")
    title = clean_text(_safe_str(data.get("title") or data.get("threadName"))) or json_file.stem.rsplit("_", 1)[0]

    participants_set = set()
    participants = data.get("participants")
    if participants and isinstance(participants, list):
        for p in participants:
            # Handle both formats: string or dict with "name" key
            if isinstance(p, str):
                name = clean_text(p)
            elif isinstance(p, dict):
                name = clean_text(_safe_str(p.get("name")))
            else:
                name = ""
            if name:
                participants_set.add(name)

    all_messages = []
    messages = data.get("messages")
    if messages and isinstance(messages, list):
        for m in messages:
            cm = parse_message(m)
            if cm:
                all_messages.append(cm)

    all_messages.sort(key=lambda x: x.timestamp_ms)
    participants_list = sorted(participants_set)

    if not title.strip() and participants_list:
        title = ", ".join(participants_list[:4]) + ("…" if len(participants_list) > 4 else "")

    return title, participants_list, all_messages


def generate_unique_thread_id(base_id: str, used_ids: set[str]) -> str:
    """Generate a unique thread ID by appending a counter if needed."""
    if base_id not in used_ids:
        return base_id

    counter = 2
    while f"{base_id}_{counter}" in used_ids:
        counter += 1

    return f"{base_id}_{counter}"


# =============================================================================
# JSON AUTO-FIX
# =============================================================================

def try_fix_json(content: str) -> tuple[str, list[str]]:
    """Attempt to fix common JSON corruption patterns.

    Uses pre-compiled patterns from the Patterns class for performance.

    Returns:
        tuple: (fixed_content, list_of_fixes_applied)
    """
    fixes = []

    # Fix 1: Missing comma between objects (common Facebook export bug)
    if Patterns.MISSING_COMMA.search(content):
        content = Patterns.MISSING_COMMA.sub(r'\1,\n\2', content)
        fixes.append("Added missing comma before sender_name")

    # Fix 2: Truncated message objects - detect incomplete objects
    if Patterns.TRUNCATED_MSG.search(content):
        content = Patterns.TRUNCATED_MSG.sub(
            lambda m: m.group(0).replace(
                '"sender_name"',
                ',\n      "content": "",\n      "is_geoblocked_for_viewer": false\n    },\n    {\n      "sender_name"',
                1
            ),
            content
        )
        fixes.append("Fixed truncated message object")

    # Fix 3: Trailing commas before closing brackets
    if Patterns.TRAILING_COMMA.search(content):
        content = Patterns.TRAILING_COMMA.sub(r'\1', content)
        fixes.append("Removed trailing commas")

    # Fix 4: Unescaped newlines in strings
    if Patterns.UNESCAPED_NEWLINE.search(content):
        content = Patterns.UNESCAPED_NEWLINE.sub(r'\1\\n\2', content)
        fixes.append("Escaped newlines in content")

    return content, fixes


def load_json_with_fixes(file_path: Path) -> tuple[dict, list[str]]:
    """Load JSON file, attempting fixes if parsing fails.

    Returns:
        tuple: (parsed_data, list_of_fixes_applied)
    """
    fixes = []
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # If UTF-8 fails, try latin-1 immediately as fallback
        try:
            content = file_path.read_text(encoding="latin-1")
            fixes.append("Used latin-1 encoding")
        except Exception:
            # If that fails too, re-raise original error or let it fail
            raise

    if not content.strip():
        raise ValueError("File is empty")

    # First try normal parsing
    try:
        return json.loads(content), fixes
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error in {file_path}: {e}")

    # Try with fixes
    fixed_content, fixes = try_fix_json(content)

    if fixes:
        try:
            data = json.loads(fixed_content)
            logger.info(f"  Auto-fixed JSON: {', '.join(fixes)}")
            return data, fixes
        except json.JSONDecodeError:
            pass

    # If all fixes failed, raise the original error (or last error)
    raise ValueError(f"Failed to parse JSON in {file_path}")


# =============================================================================
# HTML EXPORT - TEMPLATE SYSTEM
# =============================================================================

def get_template_dir() -> Path:
    """Get the templates directory path.

    Looks for templates in:
    1. ./templates/ (relative to script)
    2. ~/.config/messenger-cleaner/templates/
    3. Falls back to built-in default
    """
    script_dir = Path(__file__).parent
    local_templates = script_dir / "templates"
    if local_templates.exists():
        return local_templates

    config_templates = Path.home() / ".config" / "messenger-cleaner" / "templates"
    if config_templates.exists():
        return config_templates

    return local_templates  # Will use default if doesn't exist


@lru_cache(maxsize=4)
def load_template(template_name: str) -> str:
    """Load a template file with caching.

    Args:
        template_name: Name of the template file (e.g., 'thread.html')

    Returns:
        Template content as string, or None if not found
    """
    template_dir = get_template_dir()
    template_path = template_dir / template_name

    if template_path.exists():
        logger.debug(f"Loading external template: {template_path}")
        return template_path.read_text(encoding="utf-8")

    return None


def get_html_template() -> str:
    """Get the HTML template for thread export.

    Tries to load from external file first, falls back to built-in default.
    Uses Jinja2 if available for more powerful templating.
    """
    external = load_template("thread.html")
    if external:
        return external
    return DEFAULT_HTML_TEMPLATE


# Default built-in HTML template (used as fallback)
DEFAULT_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Messenger Archive</title>
    <style>
        :root {{
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --bg-message-self: #0084ff;
            --bg-message-other: #303030;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --border-color: #333;
            --accent: #0084ff;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
        }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
        .header {{
            background: var(--bg-secondary);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }}
        .header h1 {{ font-size: 1.8em; margin-bottom: 15px; }}
        .header .meta {{ color: var(--text-secondary); font-size: 0.9em; }}
        .header .meta span {{ margin-right: 20px; }}
        .messages {{ display: flex; flex-direction: column; gap: 8px; }}
        .message {{
            max-width: 75%;
            padding: 12px 16px;
            border-radius: 18px;
            position: relative;
        }}
        .message.self {{
            background: var(--bg-message-self);
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }}
        .message.other {{
            background: var(--bg-message-other);
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }}
        .message .sender {{
            font-size: 0.75em;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}
        .message.self .sender {{ color: rgba(255,255,255,0.7); }}
        .message .content {{ word-wrap: break-word; white-space: pre-wrap; }}
        .message .time {{
            font-size: 0.7em;
            color: var(--text-secondary);
            margin-top: 6px;
            text-align: right;
        }}
        .message.self .time {{ color: rgba(255,255,255,0.6); }}
        .message .attachments {{
            font-size: 0.8em;
            color: var(--accent);
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}
        .message .reactions {{
            font-size: 0.8em;
            margin-top: 4px;
        }}
        .date-divider {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85em;
            margin: 20px 0;
            position: relative;
        }}
        .date-divider::before, .date-divider::after {{
            content: '';
            position: absolute;
            top: 50%;
            width: 30%;
            height: 1px;
            background: var(--border-color);
        }}
        .date-divider::before {{ left: 0; }}
        .date-divider::after {{ right: 0; }}
        .search-box {{
            position: sticky;
            top: 10px;
            background: var(--bg-secondary);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            z-index: 100;
        }}
        .search-box input {{
            width: 100%;
            padding: 10px 15px;
            border: 1px solid var(--border-color);
            border-radius: 20px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 1em;
        }}
        .search-box input:focus {{ outline: none; border-color: var(--accent); }}
        .highlight {{ background: yellow; color: black; padding: 0 2px; border-radius: 2px; }}
        .collapsed {{ display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="meta">
                <span><strong>Participants:</strong> {participants}</span>
                <span><strong>Messages:</strong> {message_count:,}</span>
                <span><strong>Period:</strong> {date_range}</span>
            </div>
        </div>
        <div class="search-box">
            <input type="text" id="search" placeholder="Search messages..." onkeyup="searchMessages()">
        </div>
        <div class="messages" id="messages">
{messages_html}
        </div>
    </div>
    <script>
        function searchMessages() {{
            const query = document.getElementById('search').value.toLowerCase();
            const messages = document.querySelectorAll('.message');
            messages.forEach(msg => {{
                const content = msg.textContent.toLowerCase();
                if (query === '' || content.includes(query)) {{
                    msg.classList.remove('collapsed');
                    // Highlight matches
                    if (query) {{
                        const contentEl = msg.querySelector('.content');
                        const original = contentEl.getAttribute('data-original') || contentEl.innerHTML;
                        contentEl.setAttribute('data-original', original);
                        const regex = new RegExp('(' + query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
                        contentEl.innerHTML = original.replace(regex, '<span class="highlight">$1</span>');
                    }}
                }} else {{
                    msg.classList.add('collapsed');
                }}
            }});
        }}
    </script>
</body>
</html>'''


def write_thread_html(out_dir: Path, thread_id: str, title: str, participants: list[str],
                      msgs: list[CleanMessage], self_name: str = "Mateusz Siekierko") -> Path:
    """Export thread as styled HTML with search functionality.

    Supports both built-in template and external Jinja2 templates.
    External templates are loaded from ./templates/ or ~/.config/messenger-cleaner/templates/
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{thread_id}.html"

    messages_html_parts = []
    last_date = None

    # Pre-process messages for template rendering
    processed_messages = []
    for m in msgs:
        msg_date = m.dt_iso[:10]

        # Add date divider when date changes
        if msg_date != last_date:
            messages_html_parts.append(f'            <div class="date-divider">{msg_date}</div>')
            last_date = msg_date

        # Determine if self or other
        is_self = self_name.lower() in m.sender.lower()
        msg_class = "self" if is_self else "other"

        # Escape HTML in content
        content_escaped = html.escape(m.content) if m.content else "<em>(no text)</em>"
        content_escaped = content_escaped.replace('\n', '<br>')

        parts = [f'            <div class="message {msg_class}">']
        if not is_self:
            parts.append(f'                <div class="sender">{html.escape(m.sender)}</div>')
        parts.append(f'                <div class="content">{content_escaped}</div>')

        if m.attachments:
            parts.append(f'                <div class="attachments">📎 {html.escape(m.attachments)}</div>')
        if m.reactions:
            parts.append(f'                <div class="reactions">{html.escape(m.reactions)}</div>')

        time_str = m.dt_iso[11:16]  # HH:MM
        parts.append(f'                <div class="time">{time_str}</div>')
        parts.append('            </div>')

        messages_html_parts.append('\n'.join(parts))

        # Also store for Jinja2 template if available
        processed_messages.append({
            'date': msg_date,
            'is_self': is_self,
            'sender': m.sender,
            'content': m.content,
            'content_escaped': content_escaped,
            'attachments': m.attachments,
            'reactions': m.reactions,
            'time': time_str,
            'timestamp_ms': m.timestamp_ms,
        })

    messages_html = '\n'.join(messages_html_parts)
    date_range = f"{msgs[0].dt_iso[:10]} to {msgs[-1].dt_iso[:10]}" if msgs else "N/A"

    # Get template (external or default)
    template_str = get_html_template()

    # Use Jinja2 if available and template contains Jinja2 syntax
    if JINJA2_AVAILABLE and '{%' in template_str:
        template = Template(template_str)
        html_content = template.render(
            title=title,
            participants=participants,
            message_count=len(msgs),
            date_range=date_range,
            messages=processed_messages,
            messages_html=messages_html,
            self_name=self_name,
        )
    else:
        # Fall back to Python's str.format()
        html_content = template_str.format(
            title=html.escape(title),
            participants=html.escape(', '.join(participants)),
            message_count=len(msgs),
            date_range=date_range,
            messages_html=messages_html
        )

    html_path.write_text(html_content, encoding="utf-8")
    return html_path


# =============================================================================
# STATISTICS DASHBOARD
# =============================================================================

def generate_statistics(msgs: list[CleanMessage], title: str, participants: list[str]) -> dict:
    """Generate comprehensive statistics for a thread."""
    if not msgs:
        return {}

    stats = {
        "title": title,
        "total_messages": len(msgs),
        "participants": participants,
        "date_range": {
            "first": msgs[0].dt_iso,
            "last": msgs[-1].dt_iso,
            "days_span": (datetime.fromisoformat(msgs[-1].dt_iso[:10]) -
                         datetime.fromisoformat(msgs[0].dt_iso[:10])).days + 1
        },
        "messages_per_sender": {},
        "messages_per_month": {},
        "messages_per_weekday": defaultdict(int),
        "messages_per_hour": defaultdict(int),
        "avg_message_length": 0,
        "longest_message": {"length": 0, "sender": "", "date": ""},
        "most_active_day": {"date": "", "count": 0},
        "response_times": [],
        "attachment_stats": defaultdict(int),
        "word_count": 0,
    }

    sender_counts = Counter()
    month_counts = Counter()
    day_counts = Counter()
    total_length = 0
    word_count = 0
    prev_msg = None
    prev_sender = None

    for m in msgs:
        # Sender stats
        sender_counts[m.sender] += 1

        # Time-based stats
        dt = datetime.fromisoformat(m.dt_iso.replace('+', 'T').split('T')[0] + 'T' + m.dt_iso.split('T')[1][:8] if 'T' in m.dt_iso else m.dt_iso)
        month_key = dt.strftime("%Y-%m")
        month_counts[month_key] += 1
        stats["messages_per_weekday"][dt.strftime("%A")] += 1
        stats["messages_per_hour"][dt.hour] += 1

        day_key = dt.strftime("%Y-%m-%d")
        day_counts[day_key] += 1

        # Content stats
        content_len = len(m.content) if m.content else 0
        total_length += content_len
        word_count += len(m.content.split()) if m.content else 0

        if content_len > stats["longest_message"]["length"]:
            stats["longest_message"] = {
                "length": content_len,
                "sender": m.sender,
                "date": m.dt_iso
            }

        # Response time (simplified - time between messages from different senders)
        if prev_msg and prev_sender and prev_sender != m.sender:
            try:
                prev_dt = datetime.fromisoformat(prev_msg.dt_iso[:19])
                curr_dt = datetime.fromisoformat(m.dt_iso[:19])
                diff = (curr_dt - prev_dt).total_seconds()
                if 0 < diff < 86400:  # Less than 24 hours
                    stats["response_times"].append(diff)
            except:
                pass

        prev_msg = m
        prev_sender = m.sender

        # Attachment stats
        for detail in m.attachment_details:
            stats["attachment_stats"][detail.get("type", "unknown")] += 1

    stats["messages_per_sender"] = dict(sender_counts)
    stats["messages_per_month"] = dict(sorted(month_counts.items()))
    stats["messages_per_weekday"] = dict(stats["messages_per_weekday"])
    stats["messages_per_hour"] = dict(stats["messages_per_hour"])
    stats["avg_message_length"] = total_length / len(msgs) if msgs else 0
    stats["word_count"] = word_count
    stats["attachment_stats"] = dict(stats["attachment_stats"])

    # Most active day
    if day_counts:
        most_active = day_counts.most_common(1)[0]
        stats["most_active_day"] = {"date": most_active[0], "count": most_active[1]}

    # Average response time
    if stats["response_times"]:
        avg_response = sum(stats["response_times"]) / len(stats["response_times"])
        stats["avg_response_time_minutes"] = avg_response / 60
    else:
        stats["avg_response_time_minutes"] = None

    del stats["response_times"]  # Don't include raw data

    return stats


def write_statistics_dashboard(out_dir: Path, all_stats: list[dict]) -> Path:
    """Generate a statistics dashboard markdown file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "STATISTICS.md"

    # Aggregate stats
    total_messages = sum(s.get("total_messages", 0) for s in all_stats)
    total_words = sum(s.get("word_count", 0) for s in all_stats)

    # Combine sender stats across all threads
    all_senders = Counter()
    for s in all_stats:
        all_senders.update(s.get("messages_per_sender", {}))

    # Combine monthly stats
    all_months = Counter()
    for s in all_stats:
        all_months.update(s.get("messages_per_month", {}))

    lines = [
        "# Messenger Statistics Dashboard",
        "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        "",
        "---",
        "",
        "## Overview",
        "",
        f"- **Total Messages:** {total_messages:,}",
        f"- **Total Words:** {total_words:,}",
        f"- **Total Threads:** {len(all_stats)}",
        f"- **Avg Words/Message:** {total_words/total_messages:.1f}" if total_messages else "",
        "",
        "---",
        "",
        "## Messages by Sender (All Threads)",
        "",
        "| Sender | Messages | % |",
        "|--------|----------|---|",
    ]

    for sender, count in all_senders.most_common(20):
        pct = (count / total_messages * 100) if total_messages else 0
        lines.append(f"| {sender} | {count:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Messages by Month",
        "",
        "| Month | Messages |",
        "|-------|----------|",
    ])

    for month, count in sorted(all_months.items())[-24:]:  # Last 24 months
        lines.append(f"| {month} | {count:,} |")

    lines.extend([
        "",
        "---",
        "",
        "## Per-Thread Statistics",
        "",
        "| Thread | Messages | Words | Avg Length | Most Active Day |",
        "|--------|----------|-------|------------|-----------------|",
    ])

    for s in sorted(all_stats, key=lambda x: x.get("total_messages", 0), reverse=True):
        title = s.get("title", "Unknown")[:30]
        msg_count = s.get("total_messages", 0)
        words = s.get("word_count", 0)
        avg_len = s.get("avg_message_length", 0)
        active_day = s.get("most_active_day", {})
        active_str = f"{active_day.get('date', 'N/A')} ({active_day.get('count', 0)})" if active_day.get('date') else "N/A"
        lines.append(f"| {title} | {msg_count:,} | {words:,} | {avg_len:.0f} | {active_str} |")

    stats_path.write_text("\n".join(lines), encoding="utf-8")
    return stats_path


# =============================================================================
# SEARCH FUNCTIONALITY
# =============================================================================

def search_messages(msgs: list[CleanMessage], query: str, case_sensitive: bool = False) -> list[CleanMessage]:
    """Search messages for a query string."""
    if not case_sensitive:
        query = query.lower()

    results = []
    for m in msgs:
        content = m.content if case_sensitive else m.content.lower()
        if query in content:
            results.append(m)

    return results


def search_all_threads(thread_dirs: list[Path], query: str, case_sensitive: bool = False) -> dict[str, list[CleanMessage]]:
    """Search across all threads for a query string."""
    results = {}

    for tdir in thread_dirs:
        try:
            title, _, msgs = load_thread(tdir)
            matches = search_messages(msgs, query, case_sensitive)
            if matches:
                results[title] = matches
        except Exception:
            continue

    return results


# =============================================================================
# MERGE DUPLICATES
# =============================================================================

def find_duplicate_threads(thread_dirs: list[Path]) -> dict[str, list[Path]]:
    """Find threads that appear to be duplicates based on participants."""
    thread_signatures = defaultdict(list)

    for tdir in thread_dirs:
        try:
            title, participants, _ = load_thread(tdir)
            # Create signature from sorted participants
            sig = tuple(sorted(p.lower() for p in participants))
            if sig:
                thread_signatures[sig].append((tdir, title))
        except Exception:
            continue

    # Return only groups with duplicates
    duplicates = {}
    for sig, threads in thread_signatures.items():
        if len(threads) > 1:
            key = " + ".join(sorted(set(p.title() for p in sig)))
            duplicates[key] = [t[0] for t in threads]

    return duplicates


def merge_threads(thread_dirs: list[Path]) -> tuple[str, list[str], list[CleanMessage]]:
    """Merge multiple thread directories into one."""
    all_messages = []
    all_participants = set()
    title = ""

    for tdir in thread_dirs:
        try:
            t, participants, msgs = load_thread(tdir)
            if not title:
                title = t
            all_participants.update(participants)
            all_messages.extend(msgs)
        except Exception:
            continue

    # Deduplicate by timestamp_ms (same message in multiple exports)
    seen = set()
    unique_messages = []
    for m in all_messages:
        key = (m.timestamp_ms, m.sender, m.content[:50] if m.content else "")
        if key not in seen:
            seen.add(key)
            unique_messages.append(m)

    unique_messages.sort(key=lambda x: x.timestamp_ms)

    return title, sorted(all_participants), unique_messages


# =============================================================================
# ATTACHMENT ORGANIZER
# =============================================================================

def copy_attachments(msgs: list[CleanMessage], source_dir: Path, dest_dir: Path, thread_id: str) -> int:
    """Copy attachment files to organized output directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for m in msgs:
        for att in m.attachment_details:
            uri = att.get("uri", "")
            if not uri:
                continue

            # URI is relative to the export root
            source_path = source_dir / uri
            if not source_path.exists():
                # Try relative to thread dir
                source_path = source_dir.parent / uri
                if not source_path.exists():
                    continue

            # Organize by type
            att_type = att.get("type", "other")
            type_dir = dest_dir / thread_id / att_type
            type_dir.mkdir(parents=True, exist_ok=True)

            dest_path = type_dir / source_path.name
            if not dest_path.exists():
                try:
                    shutil.copy2(source_path, dest_path)
                    copied += 1
                except Exception as e:
                    logger.debug(f"Failed to copy {source_path}: {e}")

    return copied


# =============================================================================
# CONFIG FILE SUPPORT
# =============================================================================

DEFAULT_CONFIG = {
    "output_formats": ["md"],
    "verbose": False,
    "quiet": False,
    "parallel_workers": 4,
    "self_name": "Mateusz Siekierko",
    "copy_attachments": False,
    "generate_stats": False,
    "incremental": False,
}


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    config = DEFAULT_CONFIG.copy()

    if config_path is None:
        # Look for config in current directory or home
        for path in [Path(".cleaner.yaml"), Path.home() / ".cleaner.yaml"]:
            if path.exists():
                config_path = path
                break

    if config_path and config_path.exists():
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not installed. Config file ignored. Install with: pip install pyyaml")
            return config

        try:
            with open(config_path) as f:
                file_config = yaml.safe_load(f) or {}
            config.update(file_config)
            logger.debug(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    return config


def save_default_config(path: Path) -> None:
    """Save default configuration to YAML file."""
    if not YAML_AVAILABLE:
        logger.error("PyYAML not installed. Install with: pip install pyyaml")
        return

    with open(path, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
    logger.info(f"Saved default config to {path}")


# =============================================================================
# UNIFIED INDEX
# =============================================================================

def write_unified_index(out_dir: Path, sources: list[tuple[str, list[ThreadStats]]]) -> Path:
    """Generate a unified index combining multiple source folders."""
    idx = out_dir / "UNIFIED_INDEX.md"

    all_items = []
    for source_name, items in sources:
        for item in items:
            all_items.append((source_name, item))

    total_messages = sum(item.message_count for _, item in all_items)
    total_threads = len(all_items)

    lines = [
        "# Unified Messenger Archive Index",
        "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"**Total Sources:** {len(sources)}",
        f"**Total Threads:** {total_threads}",
        f"**Total Messages:** {total_messages:,}",
        "",
        "---",
        "",
    ]

    for source_name, items in sources:
        source_total = sum(item.message_count for item in items)
        lines.extend([
            f"## {source_name}",
            "",
            f"**Threads:** {len(items)} | **Messages:** {source_total:,}",
            "",
            "| Messages | Thread | Date Range | File |",
            "|---:|---|---|---|",
        ])

        for item in sorted(items, key=lambda x: x.message_count, reverse=True):
            safe_title = item.title.replace("|", "\\|")
            date_range = f"{item.first_date} to {item.last_date}" if item.first_date else "N/A"
            lines.append(f"| {item.message_count:,} | {safe_title} | {date_range} | [{item.thread_id}]({item.md_relpath}) |")

        lines.extend(["", "---", ""])

    idx.write_text("\n".join(lines), encoding="utf-8")
    return idx


# =============================================================================
# HASH-BASED CACHING LAYER
# =============================================================================

@dataclass
class CachedThreadData:
    """Cached processed thread data."""
    content_hash: str
    title: str
    participants: list[str]
    messages: list[CleanMessage]
    stats: Optional[ThreadStats] = None
    thread_stats: Optional[dict] = None
    cached_at: str = ""


class ThreadCache:
    """Hash-based cache for processed thread data.

    Stores processed thread data based on content hashes.
    If the source files haven't changed (same hash), the cached
    result is returned instantly without re-parsing.

    Cache location: ~/.cache/messenger-cleaner/
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "messenger-cleaner"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _compute_hash(self, thread_dir: Path) -> str:
        """Compute a hash of all JSON files in a thread directory.

        Uses SHA256 of file contents for reliable change detection.
        For large files, hashes only the first 1MB + file size for speed.
        """
        hasher = hashlib.sha256()

        json_files = iter_message_json_files(thread_dir)
        for fp in sorted(json_files):
            # Add filename to hash
            hasher.update(fp.name.encode('utf-8'))

            # Add file size
            file_size = fp.stat().st_size
            hasher.update(str(file_size).encode('utf-8'))

            # For large files, hash first 1MB only (performance)
            if file_size > 1024 * 1024:
                with open(fp, 'rb') as f:
                    hasher.update(f.read(1024 * 1024))
            else:
                hasher.update(fp.read_bytes())

        return hasher.hexdigest()[:16]

    def _get_cache_path(self, thread_dir: Path, content_hash: str) -> Path:
        """Get the cache file path for a thread."""
        thread_key = slugify(thread_dir.name)[:50]
        return self.cache_dir / f"{thread_key}_{content_hash}.pkl"

    def get(self, thread_dir: Path) -> Optional[CachedThreadData]:
        """Retrieve cached thread data if available and valid.

        Returns None if cache miss or invalid.
        """
        try:
            content_hash = self._compute_hash(thread_dir)
            cache_path = self._get_cache_path(thread_dir, content_hash)

            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)

                # Verify hash matches
                if isinstance(cached, CachedThreadData) and cached.content_hash == content_hash:
                    self._hits += 1
                    logger.debug(f"Cache hit for {thread_dir.name}")
                    return cached

        except Exception as e:
            logger.debug(f"Cache read failed for {thread_dir.name}: {e}")

        self._misses += 1
        return None

    def put(self, thread_dir: Path, title: str, participants: list[str],
            messages: list[CleanMessage], stats: Optional[ThreadStats] = None,
            thread_stats: Optional[dict] = None) -> None:
        """Store processed thread data in cache."""
        try:
            content_hash = self._compute_hash(thread_dir)
            cache_path = self._get_cache_path(thread_dir, content_hash)

            cached = CachedThreadData(
                content_hash=content_hash,
                title=title,
                participants=participants,
                messages=messages,
                stats=stats,
                thread_stats=thread_stats,
                cached_at=datetime.now().isoformat(),
            )

            with open(cache_path, 'wb') as f:
                pickle.dump(cached, f)

            logger.debug(f"Cached {thread_dir.name} ({len(messages)} messages)")

        except Exception as e:
            logger.debug(f"Cache write failed for {thread_dir.name}: {e}")

    def clear(self) -> int:
        """Clear all cached data. Returns number of files removed."""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        return count

    def get_stats(self) -> dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "entries": len(cache_files),
            "size_mb": total_size / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
        }


# Global cache instance (lazily initialized)
_thread_cache: Optional[ThreadCache] = None


def get_thread_cache() -> ThreadCache:
    """Get or create the global thread cache."""
    global _thread_cache
    if _thread_cache is None:
        _thread_cache = ThreadCache()
    return _thread_cache


# =============================================================================
# INCREMENTAL PROCESSING
# =============================================================================

def should_process_thread(thread_dir: Path, output_file: Path) -> bool:
    """Check if thread needs reprocessing based on modification times."""
    if not output_file.exists():
        return True

    output_mtime = output_file.stat().st_mtime

    # Check if any source file is newer than output
    for json_file in iter_message_json_files(thread_dir):
        if json_file.stat().st_mtime > output_mtime:
            return True

    return False


# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

@dataclass
class ProcessResult:
    """Result of processing a single thread."""
    success: bool
    thread_dir: Path
    thread_id: str = ""
    title: str = ""
    message_count: int = 0
    stats: Optional[ThreadStats] = None
    thread_stats: Optional[dict] = None
    error: str = ""
    skipped: bool = False


def process_single_thread(
    thread_dir: Path,
    out_root: Path,
    args: argparse.Namespace,
    after_date: Optional[datetime],
    before_date: Optional[datetime],
    used_ids_lock: Any = None,
    used_ids: Optional[set] = None
) -> ProcessResult:
    """Process a single thread (for parallel execution)."""
    result = ProcessResult(success=False, thread_dir=thread_dir)

    try:
        # Load with auto-fix
        json_files = iter_message_json_files(thread_dir)
        if not json_files:
            result.error = "No JSON files found"
            return result

        all_messages = []
        title = thread_dir.name
        participants_set = set()

        for fp in json_files:
            try:
                data, fixes = load_json_with_fixes(fp)
            except Exception as e:
                result.error = str(e)
                return result

            if isinstance(data, dict):
                title = clean_text(_safe_str(data.get("title"))) or title
                participants = data.get("participants") or []
                for p in participants:
                    # Handle both formats: string or dict with "name" key
                    if isinstance(p, str):
                        name = clean_text(p)
                    elif isinstance(p, dict):
                        name = clean_text(_safe_str(p.get("name")))
                    else:
                        name = ""
                    if name:
                        participants_set.add(name)

                messages = data.get("messages") or []
                for m in messages:
                    cm = parse_message(m)
                    if cm:
                        all_messages.append(cm)

        all_messages.sort(key=lambda x: x.timestamp_ms)
        participants_list = sorted(participants_set)

        if not title.strip() and participants_list:
            title = ", ".join(participants_list[:4]) + ("…" if len(participants_list) > 4 else "")

        # Date filtering
        msgs = filter_messages_by_date(all_messages, after=after_date, before=before_date)
        if not msgs:
            result.skipped = True
            result.error = "No messages in date range"
            return result

        # Generate thread ID
        base_id = slugify(title) or slugify(thread_dir.name) or "thread"
        thread_id = base_id

        if used_ids is not None:
            if used_ids_lock:
                with used_ids_lock:
                    thread_id = generate_unique_thread_id(base_id, used_ids)
                    used_ids.add(thread_id)
            else:
                thread_id = generate_unique_thread_id(base_id, used_ids)
                used_ids.add(thread_id)

        result.thread_id = thread_id
        result.title = title
        result.message_count = len(msgs)

        # Output directories
        md_out = out_root / "threads_md"
        csv_out = out_root / "threads_csv"
        json_out = out_root / "threads_json"
        html_out = out_root / "threads_html"

        # Write outputs
        md_path = write_thread_markdown(md_out, thread_id, title, participants_list, msgs)

        if getattr(args, 'csv', False):
            write_thread_csv(csv_out, thread_id, msgs)

        if getattr(args, 'json', False):
            write_thread_json(json_out, thread_id, title, participants_list, msgs)

        if getattr(args, 'html', False):
            self_name = getattr(args, 'self_name', 'Mateusz Siekierko')
            write_thread_html(html_out, thread_id, title, participants_list, msgs, self_name)

        if getattr(args, 'copy_media', False):
            media_out = out_root / "media"
            copy_attachments(msgs, thread_dir, media_out, thread_id)

        rel = md_path.relative_to(out_root).as_posix()
        result.stats = compute_thread_stats(thread_id, title, participants_list, msgs, rel)

        if getattr(args, 'stats', False):
            result.thread_stats = generate_statistics(msgs, title, participants_list)

        result.success = True

    except Exception as e:
        result.error = str(e)

    return result


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse & clean Messenger JSON logs into Markdown/CSV/JSON/HTML archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input D:\\Exports\\messages\\inbox --out D:\\Archive
  %(prog)s --input ./inbox --out ./clean --csv --json --html --verbose
  %(prog)s --input ./inbox --out ./clean --after 2023-01-01 --before 2024-01-01
  %(prog)s --input ./inbox --out ./clean --dry-run
  %(prog)s --input ./inbox --out ./clean --parallel 8 --stats
  %(prog)s --search "keyword" --input ./inbox
  %(prog)s --merge --input ./inbox ./e2ee --out ./merged
        """
    )

    # Required arguments (input required, out required for most modes)
    ap.add_argument("--input", nargs="+", required=True, help="Path(s) to messages/inbox or thread folders")
    ap.add_argument("--out", help="Output directory")

    # Output format options
    ap.add_argument("--csv", action="store_true", help="Also export per-thread CSV")
    ap.add_argument("--json", action="store_true", help="Also export per-thread JSON")
    ap.add_argument("--html", action="store_true", help="Also export per-thread HTML (styled, with search)")

    # Filtering options
    ap.add_argument("--limit", type=int, default=0, help="Limit number of threads processed (0 = no limit)")
    ap.add_argument("--after", type=str, help="Only include messages after YYYY-MM-DD")
    ap.add_argument("--before", type=str, help="Only include messages before YYYY-MM-DD")

    # Execution options
    ap.add_argument("--dry-run", action="store_true", help="List threads without processing (preview mode)")
    ap.add_argument("--incremental", action="store_true", help="Skip already-processed threads")
    ap.add_argument("--parallel", type=int, default=1, metavar="N", help="Process N threads in parallel (default: 1)")
    ap.add_argument("--use-cache", action="store_true", help="Use hash-based caching for faster re-runs")
    ap.add_argument("--clear-cache", action="store_true", help="Clear the cache and exit")
    ap.add_argument("--cache-stats", action="store_true", help="Show cache statistics and exit")
    ap.add_argument("--streaming", action="store_true", help="Use streaming JSON parser for large files (requires ijson)")

    # Advanced features
    ap.add_argument("--stats", action="store_true", help="Generate statistics dashboard")
    ap.add_argument("--copy-media", action="store_true", help="Copy attachment files to output")
    ap.add_argument("--merge", action="store_true", help="Merge duplicate threads from multiple sources")
    ap.add_argument("--search", type=str, metavar="QUERY", help="Search for keyword across all threads")
    ap.add_argument("--self-name", type=str, default="Mateusz Siekierko", help="Your name (for HTML styling)")

    # Config
    ap.add_argument("--config", type=str, help="Path to config file (.cleaner.yaml)")
    ap.add_argument("--save-config", type=str, metavar="PATH", help="Save default config to file")

    # Verbosity options
    verbosity = ap.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    verbosity.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output")

    args = ap.parse_args(argv)

    # Handle save-config
    if args.save_config:
        setup_logging(verbose=True)
        save_default_config(Path(args.save_config))
        return 0

    # Handle cache commands
    if args.clear_cache:
        setup_logging(verbose=True)
        cache = get_thread_cache()
        count = cache.clear()
        logger.info(f"Cleared {count} cached entries from {cache.cache_dir}")
        return 0

    if args.cache_stats:
        setup_logging(verbose=True)
        cache = get_thread_cache()
        stats = cache.get_stats()
        logger.info("Cache Statistics:")
        logger.info(f"  Location:    {cache.cache_dir}")
        logger.info(f"  Entries:     {stats['entries']}")
        logger.info(f"  Size:        {stats['size_mb']:.2f} MB")
        return 0

    # Load config file
    config = load_config(Path(args.config) if args.config else None)

    # Setup logging based on verbosity
    setup_logging(verbose=args.verbose or config.get("verbose", False),
                  quiet=args.quiet or config.get("quiet", False))

    # Search mode
    if args.search:
        logger.info(f"Searching for: '{args.search}'")
        all_threads = []
        for inp in args.input:
            in_path = Path(inp).expanduser().resolve()
            all_threads.extend(find_thread_dirs(in_path))

        results = search_all_threads(all_threads, args.search)
        if results:
            total_matches = sum(len(m) for m in results.values())
            logger.info(f"Found {total_matches} matches in {len(results)} thread(s):\n")
            for title, matches in sorted(results.items(), key=lambda x: len(x[1]), reverse=True):
                logger.info(f"  {title}: {len(matches)} matches")
                for m in matches[:3]:  # Show first 3 matches
                    preview = m.content[:100].replace('\n', ' ') + "..." if len(m.content) > 100 else m.content.replace('\n', ' ')
                    logger.info(f"    [{m.dt_iso[:10]}] {m.sender}: {preview}")
                if len(matches) > 3:
                    logger.info(f"    ... and {len(matches) - 3} more")
                logger.info("")
        else:
            logger.info("No matches found.")
        return 0

    # Require --out for processing modes
    if not args.out:
        logger.error("--out is required for processing. Use --search for search-only mode.")
        return 1

    # Parse date filters
    try:
        after_date = parse_date_arg(args.after)
        before_date = parse_date_arg(args.before)
    except ValueError as e:
        logger.error(str(e))
        return 4

    out_root = Path(args.out).expanduser().resolve()

    # Collect all threads from all input paths
    all_threads = []  # (path, is_standalone_file)
    input_sources = []
    for inp in args.input:
        in_path = Path(inp).expanduser().resolve()
        if not in_path.exists():
            logger.error(f"Input path not found: {in_path}")
            return 2

        # Find thread directories (folder-based format)
        threads = find_thread_dirs(in_path)
        for t in threads:
            all_threads.append((t, False))

        # Find standalone JSON files (file-based format)
        standalone_files = find_standalone_json_files(in_path)
        for f in standalone_files:
            all_threads.append((f, True))

        if threads or standalone_files:
            input_sources.append((in_path.name, threads + standalone_files))

    if not all_threads:
        logger.error("No thread folders or JSON files found. Point --input at messages/inbox or a folder with JSON files.")
        return 3

    logger.info(f"Found {len(all_threads)} conversation(s) to process")

    # Merge mode: detect and merge duplicates
    if args.merge and len(args.input) > 1:
        logger.info("Detecting duplicate threads across sources...")
        thread_paths = [t[0] for t in all_threads if not t[1]]  # Only directories
        duplicates = find_duplicate_threads(thread_paths)
        if duplicates:
            logger.info(f"Found {len(duplicates)} duplicate thread group(s)")
            for name, paths in duplicates.items():
                logger.info(f"  {name}: {len(paths)} copies")
        else:
            logger.info("No duplicates found.")

    if args.limit and args.limit > 0:
        all_threads = all_threads[: args.limit]

    total_threads = len(all_threads)

    # Helper function for progress iteration
    def progress_iter(iterable, desc: str = "Processing", disable: bool = False):
        """Wrap iterable with tqdm progress bar if available."""
        if TQDM_AVAILABLE and not disable and not args.quiet:
            return tqdm(
                iterable,
                desc=desc,
                unit="thread",
                total=len(iterable) if hasattr(iterable, '__len__') else None,
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        return iterable

    # Dry-run mode
    if args.dry_run:
        logger.info(f"Dry-run mode: Found {total_threads} conversation(s) to process")
        logger.info("")
        for i, (path, is_standalone) in enumerate(all_threads, 1):
            marker = "[FILE]" if is_standalone else "[DIR]"
            logger.info(f"  [{i}/{total_threads}] {marker} {path.name}")
        logger.info("")
        logger.info("No files were written. Remove --dry-run to process.")
        return 0

    # Processing
    index_items: list[ThreadStats] = []
    all_stats: list[dict] = []
    used_thread_ids: set[str] = set()
    processed = 0
    skipped = 0

    md_out = out_root / "threads_md"

    # Parallel processing
    if args.parallel > 1:
        import threading
        lock = threading.Lock()

        logger.info(f"Processing {total_threads} threads with {args.parallel} workers...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_single_thread,
                    tdir, out_root, args, after_date, before_date, lock, used_thread_ids
                ): tdir for tdir in all_threads
            }

            # Use tqdm for progress tracking if available
            completed_futures = concurrent.futures.as_completed(futures)
            if TQDM_AVAILABLE and not args.quiet:
                completed_futures = tqdm(
                    completed_futures,
                    total=total_threads,
                    desc="Processing (parallel)",
                    unit="thread",
                    ncols=80,
                )

            for future in completed_futures:
                result = future.result()
                if result.success:
                    index_items.append(result.stats)
                    if result.thread_stats:
                        all_stats.append(result.thread_stats)
                    processed += 1
                    if args.verbose:
                        logger.debug(f"[OK] {result.title} ({result.message_count:,} messages)")
                elif result.skipped:
                    skipped += 1
                    logger.debug(f"[SKIP] {result.thread_dir.name}: {result.error}")
                else:
                    skipped += 1
                    logger.warning(f"[FAIL] {result.thread_dir.name}: {result.error}")
    else:
        # Sequential processing with progress bar
        threads_iter = progress_iter(all_threads, desc="Processing")
        for path, is_standalone in threads_iter:
            if not TQDM_AVAILABLE or args.quiet:
                logger.info(f"Processing: {path.name}")

            # Incremental mode check
            if args.incremental:
                output_file = md_out / f"{slugify(path.stem if is_standalone else path.name)}.md"
                if output_file.exists():
                    source_mtime = path.stat().st_mtime
                    if output_file.stat().st_mtime > source_mtime:
                        logger.debug(f"  [SKIP] Already processed (incremental mode)")
                        skipped += 1
                        continue

            try:
                if is_standalone:
                    # Load standalone JSON file
                    title, participants_list, msgs = load_standalone_json(path)
                else:
                    # Use auto-fixing JSON loader for directory-based threads
                    json_files = iter_message_json_files(path)
                    if not json_files:
                        raise FileNotFoundError("No JSON files found")

                    all_messages = []
                    title = path.name
                    participants_set = set()

                    for fp in json_files:
                        data, fixes = load_json_with_fixes(fp)
                        if fixes:
                            logger.info(f"  Auto-fixed: {', '.join(fixes)}")

                        if isinstance(data, dict):
                            title = clean_text(_safe_str(data.get("title"))) or title
                            participants = data.get("participants") or []
                            for p in participants:
                                # Handle both formats: string or dict with "name" key
                                if isinstance(p, str):
                                    name = clean_text(p)
                                elif isinstance(p, dict):
                                    name = clean_text(_safe_str(p.get("name")))
                                else:
                                    name = ""
                                if name:
                                    participants_set.add(name)
                            messages = data.get("messages") or []
                            for m in messages:
                                cm = parse_message(m)
                                if cm:
                                    all_messages.append(cm)

                    all_messages.sort(key=lambda x: x.timestamp_ms)
                    participants_list = sorted(participants_set)

                    if not title.strip() and participants_list:
                        title = ", ".join(participants_list[:4]) + ("…" if len(participants_list) > 4 else "")

                    msgs = all_messages

            except Exception as e:
                logger.warning(f"  [SKIP] {path} -> {e}")
                skipped += 1
                continue

            # Apply date filtering
            original_count = len(msgs)
            msgs = filter_messages_by_date(msgs, after=after_date, before=before_date)

            if not msgs:
                logger.debug(f"  [SKIP] No messages in date range (had {original_count} before filter)")
                skipped += 1
                continue

            # Generate unique thread ID
            base_name = path.stem if is_standalone else path.name
            base_id = slugify(title) or slugify(base_name) or "thread"
            thread_id = generate_unique_thread_id(base_id, used_thread_ids)
            used_thread_ids.add(thread_id)

            # Write outputs
            md_path = write_thread_markdown(md_out, thread_id, title, participants_list, msgs)

            if args.csv:
                write_thread_csv(out_root / "threads_csv", thread_id, msgs)

            if args.json:
                write_thread_json(out_root / "threads_json", thread_id, title, participants_list, msgs)

            if args.html:
                write_thread_html(out_root / "threads_html", thread_id, title, participants_list, msgs, args.self_name)

            if args.copy_media:
                source_dir = path.parent if is_standalone else path
                copied = copy_attachments(msgs, source_dir, out_root / "media", thread_id)
                if copied:
                    logger.debug(f"  Copied {copied} attachment(s)")

            rel = md_path.relative_to(out_root).as_posix()
            stats = compute_thread_stats(thread_id, title, participants_list, msgs, rel)
            index_items.append(stats)

            if args.stats:
                thread_stats = generate_statistics(msgs, title, participants_list)
                all_stats.append(thread_stats)

            processed += 1
            logger.info(f"  [OK] {title} ({len(msgs):,} messages)")

    # Write index
    write_index(out_root, index_items)

    # Write unified index for multiple sources
    if len(input_sources) > 1:
        source_stats = []
        for source_name, threads in input_sources:
            source_items = [item for item in index_items if any(str(t.name) in item.thread_id for t in threads)]
            source_stats.append((source_name, source_items))
        write_unified_index(out_root, source_stats)
        logger.info(f"  Unified Index: {out_root / 'UNIFIED_INDEX.md'}")

    # Write statistics dashboard
    if args.stats and all_stats:
        stats_path = write_statistics_dashboard(out_root, all_stats)
        logger.info(f"  Statistics:  {stats_path}")

    # Summary
    total_messages = sum(item.message_count for item in index_items)
    logger.info("")
    logger.info("=" * 50)
    logger.info("Processing complete!")
    logger.info(f"  Processed: {processed} thread(s)")
    logger.info(f"  Messages:  {total_messages:,}")
    if skipped:
        logger.info(f"  Skipped:   {skipped} thread(s)")
    logger.info(f"  Index:     {out_root / 'INDEX.md'}")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
