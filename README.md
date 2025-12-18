# Log Defragmentator Lab

**Transform chaotic Facebook/Messenger JSON exports into clean, searchable archives.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What It Does

When you export your Facebook data, you get a mess:
- **Mojibake everywhere** — Polish `ż` becomes `Å¼`, `ą` becomes `Ä…`
- **Scattered JSON files** — `message_1.json`, `message_2.json`, etc.
- **No search** — Good luck finding that one message from 2019
- **Unreadable format** — Raw JSON with timestamps in milliseconds

**Log Defragmentator Lab** fixes all of this:

```
Facebook Export (messy)          Your Archive (clean)
├── inbox/                       ├── threads_md/
│   ├── john_doe/               │   │   ├── john_doe.md
│   │   ├── message_1.json      │   │   └── jane_smith.md
│   │   └── message_2.json      │   ├── threads_html/
│   └── jane_smith/             │   │   ├── john_doe.html  ← searchable!
│       └── message_1.json      │   │   └── jane_smith.html
└── ...                          ├── INDEX.md
                                 └── STATISTICS.md
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Encoding Fix** | Automatically repairs Facebook's UTF-8 mojibake |
| **Multiple Formats** | Export to Markdown, CSV, JSON, or styled HTML |
| **Full-Text Search** | Search across all conversations instantly |
| **Statistics** | Message counts, activity patterns, response times |
| **Progress Bar** | Visual feedback for large archives |
| **Parallel Processing** | Use multiple CPU cores for speed |
| **Streaming Parser** | Handle 500MB+ files without running out of RAM |
| **Smart Caching** | Near-instant re-runs for unchanged files |
| **Custom Templates** | Customize HTML output with Jinja2 |

---

## Quick Start

### 1. Install

```bash
# Clone or download this folder
cd log-defragmentator-lab

# Install optional dependencies (recommended)
pip install tqdm jinja2 ijson pyyaml
```

### 2. Run

```bash
# Basic usage
python log_fix.py --input /path/to/facebook/messages/inbox --out ./my_archive

# With all the bells and whistles
python log_fix.py --input ./inbox --out ./archive --html --csv --stats --parallel 4
```

### 3. Browse

Open `my_archive/INDEX.md` or any HTML file in `threads_html/` folder.

---

## Installation

### Requirements

- **Python 3.10+** (required)
- **tqdm** (optional) — progress bars
- **jinja2** (optional) — custom HTML templates
- **ijson** (optional) — streaming JSON for huge files
- **pyyaml** (optional) — config file support

```bash
# All optional dependencies
pip install tqdm jinja2 ijson pyyaml
```

---

## Usage

### Basic Commands

```bash
# Preview what will be processed (dry run)
python log_fix.py --input ./inbox --out ./archive --dry-run

# Process with Markdown output only
python log_fix.py --input ./inbox --out ./archive

# Full export (MD + CSV + JSON + HTML)
python log_fix.py --input ./inbox --out ./archive --csv --json --html

# Filter by date range
python log_fix.py --input ./inbox --out ./archive --after 2023-01-01 --before 2024-01-01
```

### Performance Options

```bash
# Parallel processing (4 threads)
python log_fix.py --input ./inbox --out ./archive --parallel 4

# Use caching for faster re-runs
python log_fix.py --input ./inbox --out ./archive --use-cache

# Streaming mode for huge files (500MB+)
python log_fix.py --input ./inbox --out ./archive --streaming
```

### Search

```bash
# Search for a keyword across all conversations
python log_fix.py --input ./inbox --search "vacation photos"
```

### Cache Management

```bash
# View cache statistics
python log_fix.py --cache-stats --input .

# Clear the cache
python log_fix.py --clear-cache --input .
```

---

## Output Formats

### Markdown (default)

Clean, readable format perfect for note-taking apps like Obsidian or Notion.

```markdown
# John Doe

**Thread ID:** `john_doe`
**Participants:** John Doe, You
**Messages:** 1,234

---

## 2024-01-15T14:30:00 — John Doe

Hey! How's it going?

---
```

### HTML

Styled, searchable interface with dark/light mode support.

- Full-text search with highlighting
- Keyboard shortcuts (`/` to search, `Esc` to clear)
- Responsive design
- Jump-to-top button

### CSV

Spreadsheet-friendly format for analysis in Excel, Google Sheets, or pandas.

```csv
dt_iso,sender,content,reactions,attachments,timestamp_ms
2024-01-15T14:30:00,John Doe,Hey! How's it going?,,1705329000000
```

### JSON

Structured format for programmatic processing.

```json
{
  "thread_id": "john_doe",
  "title": "John Doe",
  "participants": ["John Doe", "You"],
  "message_count": 1234,
  "messages": [...]
}
```

---

## Project Structure

```
log-defragmentator-lab/
├── log_fix.py           # Main script
├── templates/
│   └── thread.html      # Customizable HTML template (Jinja2)
├── docs/
│   └── EXAMPLES.md      # Detailed usage examples
├── samples/             # Sample data for testing
├── README.md            # This file
└── LICENSE              # MIT License
```

---

## Customization

### HTML Templates

Create custom HTML templates in the `templates/` folder. The script uses Jinja2 templating.

Available variables:
- `{{ title }}` — Conversation title
- `{{ participants }}` — List of participant names
- `{{ message_count }}` — Total messages
- `{{ date_range }}` — First to last message date
- `{{ messages }}` — List of message objects
- `{{ self_name }}` — Your name (for styling "self" messages)

### Config File

Create `.cleaner.yaml` in your home directory or current folder:

```yaml
output_formats:
  - md
  - html
verbose: false
parallel_workers: 4
self_name: "Your Name"
copy_attachments: false
generate_stats: true
```

---

## Troubleshooting

### "No thread folders found"

Make sure you're pointing to the correct folder. Facebook exports have this structure:
```
your_facebook_data/
└── messages/
    └── inbox/          ← Point --input here
        ├── person1/
        ├── person2/
        └── ...
```

### Mojibake not fixed

Some files may have unusual encoding. Try:
```bash
python log_fix.py --input ./inbox --out ./archive --verbose
```

### Out of memory on large files

Use streaming mode:
```bash
pip install ijson
python log_fix.py --input ./inbox --out ./archive --streaming
```

---

## Performance Tips

| Archive Size | Recommended Settings |
|--------------|---------------------|
| < 100 threads | Default settings |
| 100-500 threads | `--parallel 4` |
| 500+ threads | `--parallel 8 --use-cache` |
| Files > 100MB | `--streaming` |

---

## License

MIT License — feel free to use, modify, and distribute.

---

## Credits

Built with Python. Inspired by the frustration of trying to read Facebook's data exports.

**Optional dependencies:**
- [tqdm](https://github.com/tqdm/tqdm) — Progress bars
- [Jinja2](https://jinja.palletsprojects.com/) — Templating
- [ijson](https://github.com/ICRAR/ijson) — Streaming JSON
- [PyYAML](https://pyyaml.org/) — Config files
