#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bulk update script for GitHub project issues Work Started field.
Converts the bash run_bulk_update function to Python using asyncio and httpx.
"""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator

import httpx


# Global debug mode flag
DEBUG_MODE = os.getenv("DEBUG", "false").lower()


def info_flush(*args, **kwargs):
    """Print with immediate flush for GitHub Actions visibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


def debug_flush(*args, **kwargs):
    """Print debug messages only when DEBUG_MODE is enabled."""
    if DEBUG_MODE == "true":
        info_flush(*args, **kwargs)


class GitHubProjectUpdater:
    """Handles bulk updates of GitHub project issue fields with rate limiting."""

    def __init__(
        self,
        token: str,
        repository: str,
        project_id: str,
        work_started_field_id: str,
        max_concurrent: int = 5,
        headers: dict = None,
    ):
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        client = httpx.AsyncClient(headers=headers, base_url="https://api.github.com")
        self.repository = repository
        self.project_id = project_id
        self.work_started_field_id = work_started_field_id
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.collect_errors = []

    async def make_api_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make API request with infinite retry for rate limits, limited retry for other errors."""

        attempt = 0
        max_non_rate_limit_attempts = 5

        while True:
            attempt += 1

            try:
                async with self.semaphore:
                    debug_flush(f"   🔍 Making API request...")
                    response = await self.client.request(method, url, **kwargs)

                # Check for rate limiting - retry infinitely
                debug_flush(f"   ✅ API request response: {response.status_code}")
                if response.status_code == 403:
                    try:
                        error_data = response.json()
                        if "API rate limit exceeded" in str(error_data):
                            wait_time = 60 + random.randint(30, 120)  # 60-180 seconds
                            info_flush(
                                f"🛑 Rate limited (403): Retrying in {wait_time}s (attempt #{attempt})"
                            )
                            await asyncio.sleep(wait_time)
                            continue  # Infinite retry for rate limits
                    except Exception:
                        pass

                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        wait_time = int(retry_after) + random.randint(10, 30)
                    else:
                        wait_time = 60 + random.randint(30, 120)

                    info_flush(
                        f"🛑 Rate limited (429): Retrying in {wait_time}s (attempt #{attempt})"
                    )
                    await asyncio.sleep(wait_time)
                    continue  # Infinite retry for rate limits

                # Check for other HTTP errors - limited retry
                if response.status_code >= 400:
                    if attempt < max_non_rate_limit_attempts:
                        wait_time = (2**attempt) + random.randint(1, 10)
                        info_flush(
                            f"❌ HTTP {response.status_code}: Retrying in {wait_time}s (attempt {attempt}/{max_non_rate_limit_attempts})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        info_flush(
                            f"❌ HTTP {response.status_code} after {max_non_rate_limit_attempts} attempts: {response.text}"
                        )
                        response.raise_for_status()

                return response

            except httpx.RequestError as e:
                if attempt < max_non_rate_limit_attempts:
                    wait_time = (2**attempt) + random.randint(1, 10)
                    info_flush(
                        f"❌ Request error: {e}, retrying in {wait_time}s (attempt {attempt}/{max_non_rate_limit_attempts})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    info_flush(
                        f"❌ Request error after {max_non_rate_limit_attempts} attempts: {e}"
                    )
                    raise

    async def make_graphql_request_with_infinite_retry(
        self, query: str, variables: dict
    ) -> dict:
        """Make GraphQL request with infinite retry for rate limits."""
        payload = {"query": query, "variables": variables}

        attempt = 0
        while True:
            attempt += 1

            try:
                debug_flush(f"   🔍 Making GraphQL request...")
                response = await self.make_api_request("POST", "/graphql", json=payload)

                if response.status_code != 200:
                    info_flush(
                        f"❌ GraphQL HTTP {response.status_code}: {response.text}"
                    )
                    return {"errors": [{"message": f"HTTP {response.status_code}"}]}

                data = response.json()

                # Check for GraphQL rate limit errors - retry infinitely
                if data.get("errors"):
                    for error in data["errors"]:
                        if error.get("type") == "RATE_LIMITED":
                            wait_time = 60 + random.randint(30, 120)
                            info_flush(
                                f"🛑 GraphQL Rate Limited: Retrying in {wait_time}s (attempt #{attempt})"
                            )
                            await asyncio.sleep(wait_time)
                            continue  # Continue the retry loop for rate limit
                        else:
                            break

                return data

            except Exception as e:
                # For other exceptions, let them propagate up
                info_flush(f"❌ GraphQL request exception: {e}")
                raise

    async def update_work_started_field(self, item_id: str, date_value: str) -> dict:
        """Update Work Started field for a project item."""

        # Update the field
        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: ProjectV2FieldValue!) {
            updateProjectV2ItemFieldValue(input: {
                projectId: $projectId,
                itemId: $itemId,
                fieldId: $fieldId,
                value: $value
            }) {
                clientMutationId
            }
        }
        """

        variables = {
            "projectId": self.project_id,
            "itemId": item_id,
            "fieldId": self.work_started_field_id,
            "value": {"date": date_value},
        }

        return await self.make_graphql_request_with_infinite_retry(mutation, variables)

    async def process_issue(self, issue_number: int) -> None:
        """Process a single issue: check status and update Work Started if needed."""
        info_flush(
            f"🔄 Processing repository {self.repository} issue #{issue_number}..."
        )

        # Step 1: Get issue status and work started
        result = await self.get_issue_status_and_work_started(issue_number)
        if result.get("errors"):
            self.collect_errors.append(
                f"   ❌ get_issue_status_and_work_started Errors: {result['errors']}"
            )
            return

        issue_status = result["status"]
        issue_work_started = result["work_started"]
        item_id = result["item_id"]
        updated_at = result["updated_at"]
        info_flush(
            f"   📊 Status: '{issue_status}', Work Started: '{issue_work_started}', Updated At: '{updated_at}'"
        )

        # Step 2: Update Work Started if needed (matching workflow logic exactly)
        # Condition: Status is "In Progress" AND Work Started is empty/null
        if issue_status == "In Progress" and (
            not issue_work_started or issue_work_started == "null"
        ):
            info_flush(
                f"   🔧 repository {self.repository} issue #{issue_number} Updating Work Started field..."
            )

            # Use the date when status changed to 'In Progress', with fallback to current date (matching workflow)
            if updated_at and updated_at != "null":
                # Extract date part from timestamp (YYYY-MM-DD)
                work_started_date = updated_at.split("T")[0]
                info_flush(
                    f"   📅 Using status change date: {work_started_date} (from timestamp: {updated_at})"
                )
            else:
                # Fallback to current date
                work_started_date = time.strftime("%Y-%m-%d")
                info_flush(
                    f"   📅 Could not determine status change date, using current date: {work_started_date}"
                )

            success = await self.update_work_started_field(item_id, work_started_date)

            if success:
                info_flush(
                    f"   ✅ Updated Work Started for repository {self.repository} issue #{issue_number} to {work_started_date} (date when status changed)"
                )
            else:
                raise Exception(f"Failed to update issue #{issue_number}")
        else:
            info_flush(
                f"   ⏭️ Issue #{issue_number}: Status='{issue_status}', Work Started='{issue_work_started}' - no update needed"
            )

    async def get_all_repository_issues(self) -> AsyncGenerator[list, None]:
        """
        Fetch all open issues from repositories using pagination.

        Yields:
            list_of_issue_numbers
        """

        all_issue_numbers = 0

        for page in range(1, 51):  # Max 50 pages like workflow
            info_flush(f"   Fetching page {page} of issues...")
            url = f"/repos/{self.repository}/issues"
            params = {"state": "open", "per_page": 100, "page": page}

            response = await self.make_api_request("GET", url, params=params)

            issues = response.json()
            page_issue_count = len(issues)

            if page_issue_count == 0:
                info_flush(f"   ✅ No more issues on page {page}")
                break

            page_issue_numbers = [issue["number"] for issue in issues]
            info_flush(f"   ✅ Found {page_issue_count} issues on page {page}")
            yield page_issue_numbers
            all_issue_numbers += page_issue_count

            if page_issue_count < 100:  # Last page
                info_flush(f"   ✅ Last page reached")
                break

        info_flush(f"✅ Found {all_issue_numbers} total issues in {self.repository}")

    async def get_issue_status_and_work_started(self, issue_number: int):
        query = """
        query($issueNumber: Int!, $repo: String!, $owner: String!) {
        repository(owner: $owner, name: $repo) {
            issue(number: $issueNumber) {
            projectItems(first: 10) {
                nodes {
                ... on ProjectV2Item {
                    id
                    project {
                        id
                    }
                    workStarted: fieldValueByName(name: "Work Started") {
                    ... on ProjectV2ItemFieldDateValue {
                        date
                        updatedAt
                    }
                    }
                    status: fieldValueByName(name: "Status") {
                    ... on ProjectV2ItemFieldSingleSelectValue {
                        name
                        optionId
                        updatedAt
                    }
                    }
                }
                }
            }
            }
        }
        }
        """

        variables = {
            "issueNumber": issue_number,
            "repo": self.repository.split("/")[1],
            "owner": self.repository.split("/")[0],
        }

        debug_flush(f"   🔍 Querying issue #{issue_number}...")

        data = await self.make_graphql_request_with_infinite_retry(query, variables)

        info_flush(
            f"   ✅ get_issue_status_and_work_started: issue #{issue_number} Query response: {data}"
        )

        if data.get("errors"):
            return data
        project_items = data["data"]["repository"]["issue"]["projectItems"]["nodes"]
        # check if nodes are empty
        if not project_items:
            return {
                "errors": {
                    "message": f"No project items found for issue #{issue_number}",
                    "data": data,
                },
            }

        # Filter to find the item from our specific project
        project_item = None
        for item in project_items:
            if item["project"]["id"] == self.project_id:
                project_item = item
                break

        if not project_item:
            return {
                "errors": {
                    "message": f"project {self.project_id} not found for issue #{issue_number}",
                    "data": data,
                },
            }

        work_started = (
            project_item["workStarted"]["date"]
            if project_item["workStarted"]
            else "null"
        )
        status = project_item["status"]["name"] if project_item["status"] else "null"
        item_id = project_item["id"]
        updated_at = (
            project_item["status"]["updatedAt"] if project_item["status"] else "null"
        )

        return {
            "work_started": work_started,
            "status": status,
            "item_id": item_id,
            "updated_at": updated_at,
        }


async def main():
    """Main function that orchestrates the bulk update process."""
    info_flush("🚀 Starting GitHub Project Issue Bulk Update")
    info_flush("=" * 60)

    # Configuration from environment variables
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        info_flush("❌ ERROR: GITHUB_TOKEN environment variable is required")
        sys.exit(1)

    repositories = [
        "tenstorrent/tt-blacksmith",
        "tenstorrent/tt-mlir",
        "tenstorrent/tt-forge-fe",
        "tenstorrent/tt-forge",
        "tenstorrent/tt-xla",
        "tenstorrent/tt-torch",
    ]
    project_id = os.getenv("project_id")
    work_started_field_id = os.getenv("work_started_field_id")
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "5"))

    if not project_id:
        info_flush("❌ ERROR: PROJECT_ID environment variable is required")
        sys.exit(1)

    if not work_started_field_id:
        info_flush("❌ ERROR: WORK_STARTED_FIELD_ID environment variable is required")
        sys.exit(1)

    info_flush(f"🔧 Configuration:")
    info_flush(f"   Repositories: {repositories}")
    info_flush(f"   Project ID: {project_id}")
    info_flush(f"   Work Started Field ID: {work_started_field_id}")
    info_flush(f"   Max Concurrent: {max_concurrent}")
    info_flush(f"   Debug Mode: {DEBUG_MODE}")

    max_concurrent = 5
    info_flush(
        f"⚡ Starting concurrent processing with max {max_concurrent} simultaneous requests..."
    )
    start_time = time.time()

    collect_repo_errors = {}
    for repo_name in repositories:
        # Create updater instance for this repository
        updater = GitHubProjectUpdater(
            token=token,
            repository=repo_name,
            project_id=project_id,
            work_started_field_id=work_started_field_id,
            max_concurrent=max_concurrent,
        )
        async for repo_issue_numbers in updater.get_all_repository_issues():
            # Build coroutine tasks
            tasks = [
                updater.process_issue(issue_number)
                for issue_number in repo_issue_numbers
            ]

            # Process all coroutine tasks
            await asyncio.gather(*tasks)

        collect_repo_errors[repo_name] = updater.collect_errors

    for repo_name, errors in collect_repo_errors.items():
        info_flush(f"\n\n❌ {repo_name} Errors:")
        for error in errors:
            info_flush(f"\t{error}")
        info_flush(f"--------------------------------")


if __name__ == "__main__":
    DEBUG_MODE = "false"
    asyncio.run(main())
