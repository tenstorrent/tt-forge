# Bulk Update Issues Script Usage

This Python script (`bulk_update_issues.py`) bulk updates the Work Started field for GitHub project issues across multiple Tenstorrent repositories using asyncio and httpx for concurrent processing.

## Features

- **Asyncio-based**: Uses Python asyncio for concurrent processing
- **Rate limiting**: Implements exponential backoff when hitting GitHub API rate limits
- **Semaphore control**: Limits concurrent requests to 5 (configurable)
- **Error handling**: Robust error handling and retry logic with error collection per repository
- **Multi-repository support**: Processes issues across multiple Tenstorrent repositories
- **Project filtering**: Filters project items to match the specific project ID

## Dependencies

Install the required dependencies:

```bash
pip install -r requirements-bulk-update.txt
```

Or install httpx directly:

```bash
pip install httpx>=0.24.0
```

## Usage

1. **Set environment variables:**
   ```bash
   export GITHUB_TOKEN="your_github_token_here"
   export project_id="your_project_id_here"
   export work_started_field_id="your_work_started_field_id_here"
   export MAX_CONCURRENT="5"  # optional, defaults to 5
   export DEBUG="false"  # optional, set to "true" for debug output
   ```

2. **Run the script:**
   ```bash
   ./bulk_update_issues.py
   ```

The script will automatically process all open issues from the configured repositories.

## How it works

The script:

1. **Repository Processing:**
   - Iterates through multiple configured repositories
   - Fetches all open issues from each repository using GitHub REST API pagination
   - Processes up to 50 pages (5000 issues) per repository

2. **Issue Processing (with max 5 concurrent):**
   - Gets issue details and project items using GraphQL API
   - Filters project items to match the specified project ID
   - Checks the Status and Work Started fields for each relevant project item
   - Updates Work Started field if Status is "In Progress" and Work Started is empty
   - Uses the status update timestamp when available, falls back to current date

3. **Error Handling:**
   - Collects errors per repository for final reporting
   - Uses exponential backoff when rate limited
   - Handles missing project items gracefully
   - Continues processing other issues if individual issues fail

## Key Components

### Async Generator for Repository Issues

The script includes `get_all_repository_issues()` - an async generator method that:
- Fetches all open issues from a GitHub repository with pagination
- Yields batches of issue numbers for efficient processing
- Handles GitHub API pagination automatically (up to 50 pages / 5000 issues per repository)
- Processes multiple repositories in sequence

### Multi-Repository Processing

The script processes issues from multiple Tenstorrent repositories:
- `tenstorrent/tt-blacksmith`
- `tenstorrent/tt-mlir`
- `tenstorrent/tt-forge-fe`
- `tenstorrent/tt-forge`
- `tenstorrent/tt-xla`
- `tenstorrent/tt-torch`

## Configuration

The script requires these environment variables:
- **GITHUB_TOKEN**: Your GitHub personal access token with project and repository access
- **project_id**: GitHub project ID (e.g., `PVT_kwDOA9MHEM4AjeTl`)
- **work_started_field_id**: Work Started field ID (e.g., `PVTF_lADOA9MHEM4AjeTlzgzZQtk`)

Optional environment variables:
- **MAX_CONCURRENT**: Maximum concurrent requests (defaults to 5)
- **DEBUG**: Set to "true" for debug output (defaults to "false")

## Output Example

```
🚀 Starting GitHub Project Issue Bulk Update
============================================================
🔧 Configuration:
   Repositories: ['tenstorrent/tt-blacksmith', 'tenstorrent/tt-mlir', 'tenstorrent/tt-forge-fe', 'tenstorrent/tt-forge', 'tenstorrent/tt-xla', 'tenstorrent/tt-torch']
   Project ID: PVT_kwDOA9MHEM4AjeTl
   Work Started Field ID: PVTF_lADOA9MHEM4AjeTlzgzZQtk
   Max Concurrent: 5
   Debug Mode: false
⚡ Starting concurrent processing with max 5 simultaneous requests...

   Fetching page 1 of issues...
   ✅ Found 100 issues on page 1
   Fetching page 2 of issues...
   ✅ Found 23 issues on page 2
   ✅ Last page reached
✅ Found 123 total issues in tenstorrent/tt-mlir

🔄 Processing repository tenstorrent/tt-mlir issue #1234...
   📊 Status: 'In Progress', Work Started: 'null', Updated At: '2024-01-15T10:30:00Z'
   🔧 repository tenstorrent/tt-mlir issue #1234 Updating Work Started field...
   📅 Using status change date: 2024-01-15 (from timestamp: 2024-01-15T10:30:00Z)
   ✅ Updated Work Started for repository tenstorrent/tt-mlir issue #1234 to 2024-01-15 (date when status changed)

🔄 Processing repository tenstorrent/tt-mlir issue #1235...
   📊 Status: 'Done', Work Started: '2024-01-10', Updated At: '2024-01-10T14:20:00Z'
   ⏭️ Issue #1235: Status='Done', Work Started='2024-01-10' - no update needed

❌ tenstorrent/tt-mlir Errors:
	   ❌ get_issue_status_and_work_started Errors: [{'message': 'No project items found for issue #1236'}]
--------------------------------
```

## Rate Limiting

The script handles GitHub API rate limits automatically:
- Detects rate limit responses (403, 429)
- Uses exponential backoff (120-180 seconds base + exponential factor)
- Automatically retries failed requests
- Respects retry-after headers when provided
