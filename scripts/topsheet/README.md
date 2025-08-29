# TopSheet Automation

Google Apps Script automation for synchronizing TT-Forge project issues with Google Sheets TopSheet.

## Overview

This automation script fetches qualifying issues from the TT-Forge GitHub project and creates structured Google Sheets for project management and milestone tracking.

## Features

- **Automatic Issue Sync**: Fetches issues from TT-Forge project via GitHub GraphQL API
- **Smart Filtering**: Includes issues with "Top Problems/Issues" label OR non-empty "Top Sheet" field
- **Dual Sheet Creation**: 
  - `TT-Forge Top Issues`: Complete issue listing with custom fields
  - `Key Milestones`: Organized milestone and problem tracking
- **Custom Menu**: Adds "TT-Forge Sync" menu to Google Sheets
- **Rich Formatting**: Color coding, hyperlinks, and structured layouts

## Setup Instructions

1. **Open Google Apps Script**: Go to [script.google.com](https://script.google.com)
2. **Create New Project**: Click "New Project"
3. **Copy Code**: Copy the contents of `topsheet-automation.gs` into the script editor
4. **Configure Token**: 
   - Go to Project Settings → Script Properties
   - Add property: `GITHUB_TOKEN` with your GitHub Personal Access Token
5. **Set Permissions**: Authorize Google Sheets and UrlFetch permissions
6. **Attach to Sheet**: Run from the script editor or use the custom menu in your Google Sheet

## Usage

### Manual Execution
- Open your Google Sheet
- Click "TT-Forge Sync" → "Update Topsheet"
- Wait for completion notification

### Script Editor
- Run the `main()` function directly from Apps Script editor

## Data Sources

- **GitHub Project**: TT-Forge project (`PVT_kwDOA9MHEM4AjeTl`)
- **Qualifying Criteria**:
  - Issues with "Top Problems/Issues" label
  - Issues with non-empty "Top Sheet" custom field
- **Custom Fields**:
  - Estimated Completion Date
  - Status Update  
  - Top Sheet

## Output Sheets

### TT-Forge Top Issues
Complete listing with columns:
- Issue Number, Title, Repository, State
- Assigned To, Created At, Labels
- Custom Fields (Estimated Completion, Status Update, Top Sheet)
- Qualifying Reason, Issue URL

### Key Milestones  
Organized by Top Sheet value with two sections:
- **Key Upcoming Milestones/Deliverables**: From non-labeled issues
- **Top Problems/Issues**: From labeled issues

## Status Color Coding

- **[G]** prefix: Green background (Good/On Track)
- **[Y]** prefix: Yellow background (Warning/At Risk)  
- **[R]** prefix: Red background (Critical/Blocked)
- **[D]** prefix: Blue background (Done/Complete)

## Technical Details

- **Language**: Google Apps Script (JavaScript)
- **APIs**: GitHub GraphQL API
- **Permissions**: Google Sheets, UrlFetch
- **Rate Limiting**: Handles pagination for large datasets (50 pages max)
- **Error Handling**: Graceful failure with user notifications

## Security Notes

- GitHub token is stored securely in Script Properties
- No hardcoded credentials in the script
- API calls are authenticated via Bearer token

## Author

Created for TT-Forge project management automation.

## Status

✅ **Implemented and Working** - Currently deployed in Google Apps Script
