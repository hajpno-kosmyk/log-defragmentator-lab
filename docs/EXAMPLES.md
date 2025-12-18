# Log Fix — Usage Examples

Detailed examples for common use cases.

---

## Table of Contents

1. [Basic Usage](#1-basic-usage)
2. [Export Formats](#2-export-formats)
3. [Filtering Messages](#3-filtering-messages)
4. [Performance Optimization](#4-performance-optimization)
5. [Search Operations](#5-search-operations)
6. [Working with Multiple Sources](#6-working-with-multiple-sources)
7. [Cache Management](#7-cache-management)
8. [Statistics & Analytics](#8-statistics--analytics)
9. [Custom Templates](#9-custom-templates)
10. [Real-World Scenarios](#10-real-world-scenarios)

---

## 1. Basic Usage

### Preview Mode (Dry Run)

See what will be processed without writing any files:

```bash
python log_fix.py --input D:\Facebook\messages\inbox --out D:\Archive --dry-run
```

Output:
```
Dry-run mode: Found 47 conversation(s) to process

  [1/47] [DIR] john_doe_abc123
  [2/47] [DIR] jane_smith_def456
  [3/47] [DIR] family_group_ghi789
  ...

No files were written. Remove --dry-run to process.
```

### Simple Processing

Convert all conversations to Markdown:

```bash
python log_fix.py --input ./inbox --out ./archive
```

### Verbose Mode

See detailed progress:

```bash
python log_fix.py --input ./inbox --out ./archive --verbose
```

### Quiet Mode

Suppress all output except errors:

```bash
python log_fix.py --input ./inbox --out ./archive --quiet
```

---

## 2. Export Formats

### Markdown Only (Default)

```bash
python log_fix.py --input ./inbox --out ./archive
```

Creates:
```
archive/
├── INDEX.md
└── threads_md/
    ├── john_doe.md
    └── jane_smith.md
```

### HTML (Searchable, Styled)

```bash
python log_fix.py --input ./inbox --out ./archive --html
```

Creates:
```
archive/
├── INDEX.md
├── threads_md/
│   └── ...
└── threads_html/
    ├── john_doe.html
    └── jane_smith.html
```

### CSV (For Spreadsheets)

```bash
python log_fix.py --input ./inbox --out ./archive --csv
```

Creates:
```
archive/
├── INDEX.md
├── threads_md/
│   └── ...
└── threads_csv/
    ├── john_doe.csv
    └── jane_smith.csv
```

### JSON (For Programming)

```bash
python log_fix.py --input ./inbox --out ./archive --json
```

### All Formats

```bash
python log_fix.py --input ./inbox --out ./archive --csv --json --html
```

---

## 3. Filtering Messages

### By Date Range

Only messages from 2023:

```bash
python log_fix.py --input ./inbox --out ./archive --after 2023-01-01 --before 2024-01-01
```

Only messages after a specific date:

```bash
python log_fix.py --input ./inbox --out ./archive --after 2023-06-15
```

Only messages before a specific date:

```bash
python log_fix.py --input ./inbox --out ./archive --before 2022-12-31
```

### Limit Number of Threads

Process only the first 10 threads (useful for testing):

```bash
python log_fix.py --input ./inbox --out ./archive --limit 10
```

---

## 4. Performance Optimization

### Parallel Processing

Use 4 CPU cores:

```bash
python log_fix.py --input ./inbox --out ./archive --parallel 4
```

Use 8 CPU cores:

```bash
python log_fix.py --input ./inbox --out ./archive --parallel 8
```

### Caching for Re-runs

First run (creates cache):

```bash
python log_fix.py --input ./inbox --out ./archive --use-cache
```

Second run (uses cache, much faster):

```bash
python log_fix.py --input ./inbox --out ./archive --use-cache
```

### Streaming Mode for Large Files

For conversations with 500MB+ JSON files:

```bash
# First install ijson
pip install ijson

# Then run with streaming
python log_fix.py --input ./inbox --out ./archive --streaming
```

### Incremental Processing

Skip already-processed threads:

```bash
python log_fix.py --input ./inbox --out ./archive --incremental
```

### Combined Performance Options

For very large archives (1000+ threads):

```bash
python log_fix.py --input ./inbox --out ./archive \
    --parallel 8 \
    --use-cache \
    --streaming \
    --incremental
```

---

## 5. Search Operations

### Basic Search

Find messages containing "vacation":

```bash
python log_fix.py --input ./inbox --search "vacation"
```

Output:
```
Searching for: 'vacation'
Found 23 matches in 5 thread(s):

  Family Group: 12 matches
    [2023-07-15] Mom: Are we still going on vacation next week?
    [2023-07-15] Dad: Yes! I booked the hotel...
    [2023-07-16] You: Can't wait for this vacation!
    ... and 9 more

  John Doe: 8 matches
    [2023-06-01] John: Remember our vacation in Italy?
    ... and 7 more
```

### Search with Special Characters

Quotes and special characters work:

```bash
python log_fix.py --input ./inbox --search "what's up"
python log_fix.py --input ./inbox --search "meeting @3pm"
```

---

## 6. Working with Multiple Sources

### Multiple Input Folders

Process both regular and encrypted exports:

```bash
python log_fix.py --input ./inbox ./e2ee_inbox --out ./archive
```

### Merge Duplicate Threads

Detect and merge conversations that appear in multiple exports:

```bash
python log_fix.py --input ./inbox ./e2ee_inbox --out ./archive --merge
```

Output:
```
Detecting duplicate threads across sources...
Found 3 duplicate thread group(s)
  John Doe: 2 copies
  Jane Smith: 2 copies
  Family Group: 2 copies
```

---

## 7. Cache Management

### View Cache Statistics

```bash
python log_fix.py --cache-stats --input .
```

Output:
```
Cache Statistics:
  Location:    C:\Users\You\.cache\messenger-cleaner
  Entries:     47
  Size:        12.34 MB
```

### Clear Cache

```bash
python log_fix.py --clear-cache --input .
```

Output:
```
Cleared 47 cached entries from C:\Users\You\.cache\messenger-cleaner
```

---

## 8. Statistics & Analytics

### Generate Statistics Dashboard

```bash
python log_fix.py --input ./inbox --out ./archive --stats
```

Creates `STATISTICS.md` with:
- Total messages and words
- Messages by sender
- Messages by month
- Most active days
- Average message length

### Example Statistics Output

```markdown
# Messenger Statistics Dashboard

**Generated:** 2024-01-15T10:30:00
**Total Messages:** 45,678
**Total Words:** 234,567
**Total Threads:** 47

## Messages by Sender (All Threads)

| Sender | Messages | % |
|--------|----------|---|
| You | 23,456 | 51.3% |
| John Doe | 12,345 | 27.0% |
| Jane Smith | 9,877 | 21.6% |

## Messages by Month

| Month | Messages |
|-------|----------|
| 2024-01 | 1,234 |
| 2023-12 | 2,345 |
| 2023-11 | 1,876 |
```

---

## 9. Custom Templates

### Using Custom HTML Template

1. Edit `templates/thread.html`
2. Run normally — the custom template is auto-detected:

```bash
python log_fix.py --input ./inbox --out ./archive --html
```

### Template Variables

Available in your Jinja2 template:

```jinja2
{{ title }}           — Conversation title
{{ participants }}    — List of names
{{ message_count }}   — Number of messages
{{ date_range }}      — "2023-01-01 to 2024-01-15"
{{ messages }}        — List of message objects
{{ self_name }}       — Your name

{% for msg in messages %}
  {{ msg.date }}           — "2024-01-15"
  {{ msg.time }}           — "14:30"
  {{ msg.sender }}         — "John Doe"
  {{ msg.content }}        — Raw message content
  {{ msg.content_escaped }} — HTML-escaped content
  {{ msg.is_self }}        — true/false
  {{ msg.attachments }}    — "2 photo(s)"
  {{ msg.reactions }}      — "❤️ by Jane"
{% endfor %}
```

### Set Your Name for Self-Detection

```bash
python log_fix.py --input ./inbox --out ./archive --html --self-name "Mateusz Kowalski"
```

---

## 10. Real-World Scenarios

### Scenario A: First-Time Archive

You just downloaded your Facebook data and want a complete archive:

```bash
# Step 1: Preview
python log_fix.py --input ~/Downloads/facebook-data/messages/inbox --out ~/Documents/MessengerArchive --dry-run

# Step 2: Full export with everything
python log_fix.py --input ~/Downloads/facebook-data/messages/inbox --out ~/Documents/MessengerArchive \
    --html --csv --stats --parallel 4

# Step 3: Open the archive
# → Open ~/Documents/MessengerArchive/INDEX.md
# → Or browse threads_html/ for searchable HTML
```

### Scenario B: Monthly Backup Update

You download fresh data monthly and want to update your archive:

```bash
python log_fix.py --input ~/Downloads/facebook-2024-02/messages/inbox \
    --out ~/Documents/MessengerArchive \
    --html --use-cache --incremental
```

### Scenario C: Finding Old Messages

You need to find messages about "contract" from 2022:

```bash
# Quick search
python log_fix.py --input ~/Documents/MessengerArchive/threads_md --search "contract"

# Or filter by date and export
python log_fix.py --input ./inbox --out ./contract_search \
    --after 2022-01-01 --before 2023-01-01 --html
# Then use Ctrl+F in the HTML files
```

### Scenario D: Sharing a Single Conversation

Export just one conversation as a nice HTML file:

```bash
python log_fix.py --input ./inbox/john_doe_abc123 --out ./john_archive --html
```

### Scenario E: Large Archive (1000+ Conversations)

```bash
# Install performance dependencies
pip install tqdm ijson

# Run with all optimizations
python log_fix.py --input ./inbox --out ./archive \
    --parallel 8 \
    --use-cache \
    --streaming \
    -q  # Quiet mode, rely on progress bar

# Check results
python log_fix.py --cache-stats --input .
```

---

## Command Reference

| Flag | Short | Description |
|------|-------|-------------|
| `--input` | | Input folder(s) (required) |
| `--out` | | Output folder |
| `--csv` | | Export CSV format |
| `--json` | | Export JSON format |
| `--html` | | Export HTML format |
| `--limit N` | | Process only N threads |
| `--after DATE` | | Only messages after YYYY-MM-DD |
| `--before DATE` | | Only messages before YYYY-MM-DD |
| `--dry-run` | | Preview without processing |
| `--incremental` | | Skip already-processed |
| `--parallel N` | | Use N parallel workers |
| `--use-cache` | | Enable hash-based caching |
| `--clear-cache` | | Clear cache and exit |
| `--cache-stats` | | Show cache stats and exit |
| `--streaming` | | Use streaming JSON parser |
| `--stats` | | Generate statistics dashboard |
| `--copy-media` | | Copy attachment files |
| `--merge` | | Merge duplicate threads |
| `--search QUERY` | | Search for keyword |
| `--self-name NAME` | | Your name (for HTML) |
| `--config FILE` | | Config file path |
| `--save-config FILE` | | Save default config |
| `--verbose` | `-v` | Detailed output |
| `--quiet` | `-q` | Suppress output |

---

## Tips & Tricks

1. **Start with `--dry-run`** to see what will be processed
2. **Use `--limit 5`** when testing new options
3. **Enable `--use-cache`** if you'll run the script multiple times
4. **Use `--parallel`** with the number of CPU cores you have
5. **HTML files have built-in search** — press `/` to focus search box
6. **The INDEX.md** file lists all threads sorted by message count
