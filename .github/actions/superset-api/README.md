# Superset API Call GitHub Action

This GitHub Action allows you to retrieve data from a Postgres database via the Tenstorrent Superset API using predefined queries with customizable parameters.

## Description

The Superset API Call action provides a simple interface to query data from the Tenstorrent Postgres database through the Superset API. It handles the authentication and request signing process required to access the API Gateway endpoint using AWS SigV4 authentication.

This action enables workflows to:
- Retrieve benchmark data and metrics
- Access predefined database queries with custom parameters
- Get formatted JSON responses for further processing

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `query` | API path to query (added after base URL) | Yes | - |
| `query_params` | Query parameters as a JSON string | Yes | - |

## Outputs

| Output | Description |
|--------|-------------|
| `response` | JSON response from the API |

## Usage

### Basic Example

```yaml
- name: Call Superset API
  id: api-call
  uses: ./.github/actions/superset-api
  with:
    query: 'benchmarks/last_measurement'
    query_params: '{"project":"tenstorrent/tt-metal","ml_model_name":"tiiuae/falcon-7b-instruct"}'
```

### Example with Formatted Output

```yaml
- name: Call Superset API
  id: api-call
  uses: ./.github/actions/superset-api
  with:
    query: 'benchmarks/last_measurement'
    query_params: '{"project":"tenstorrent/tt-metal","ml_model_name":"tiiuae/falcon-7b-instruct"}'

- name: Display Response
  run: |
    echo "API Response:"
    echo '${{ steps.api-call.outputs.response }}'

- name: Create Job Summary
  run: |
    echo "# Benchmark Results" >> $GITHUB_STEP_SUMMARY
    echo "" >> $GITHUB_STEP_SUMMARY
    echo "| Metric | Value |" >> $GITHUB_STEP_SUMMARY
    echo "| ------ | ----- |" >> $GITHUB_STEP_SUMMARY

- name: Add Table Rows to Summary
  uses: actions/github-script@v6
  with:
    script: |
      const response = ${{ fromJSON(steps.api-call.outputs.response) }};
      let tableContent = '';

      for (const item of response) {
        tableContent += `| ${item.name} | ${item.last_value} |\n`;
      }

      require('fs').appendFileSync(process.env.GITHUB_STEP_SUMMARY, tableContent);
```

## Requirements

This action requires:

1. AWS credentials configured with appropriate permissions to access the API Gateway
2. Python 3.x with boto3 and requests libraries (installed automatically by the action)
3. Proper workflow permissions for AWS authentication

## Authentication & Permissions

The action uses the AWS SigV4 authentication method to sign requests to the API Gateway. It requires AWS credentials to be available in the environment, which are configured using the `aws-actions/configure-aws-credentials` action.

### Required Workflow Permissions

You must add the following permissions to your workflow to allow AWS identity-based authentication:

```yaml
permissions:
  id-token: write   # Required for OIDC authentication
  contents: read    # Required for checking out code
```

Without these permissions, the API calls will fail with authentication errors.

## API Endpoint

The action uses the following base URL for API requests:
```
https://ftuxhxqka0.execute-api.us-east-2.amazonaws.com/api-gw-data-db-main/api/v1/data_db_main/
```

The `query` input is appended to this base URL to form the complete API endpoint.
