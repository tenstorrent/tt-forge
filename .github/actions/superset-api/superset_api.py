#!/usr/bin/env python3
import boto3
import requests
import json
import sys
import os
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

def call_superset_api(query, query_params_json):
    """
    Call the Superset API and return the JSON response
    
    Args:
        query (str): API path to query (added after base URL)
        query_params_json (str): Query parameters as a JSON string
    
    Returns:
        dict: JSON response from the API
    """
    # Base API Gateway endpoint
    base_url = 'https://ftuxhxqka0.execute-api.us-east-2.amazonaws.com/api-gw-data-db-main/api/v1/data_db_main/'
    
    # Parse query parameters
    query_params = json.loads(query_params_json)
    
    # Construct full URL
    url = base_url + query
    
    # Create a request
    request = AWSRequest(
        method='GET',
        url=url,
        params=query_params
    )
    
    # Get credentials from the environment
    session = boto3.Session()
    credentials = session.get_credentials()
    
    # Sign the request
    SigV4Auth(credentials, 'execute-api', 'us-east-2').add_auth(request)
    
    # Get the signed headers
    signed_headers = dict(request.headers)
    
    # Make the request
    response = requests.get(
        url,
        params=query_params,
        headers=signed_headers
    )
    
    # Check if the request was successful
    response.raise_for_status()
    
    # Print the status code for debugging
    print(f'Status Code: {response.status_code}')
    
    # Return the response as JSON
    return response.json()

if __name__ == "__main__":
    # Get inputs from environment variables
    query = os.environ.get('INPUT_QUERY')
    query_params_json = os.environ.get('INPUT_QUERY_PARAMS')
    
    # Call the API
    result = call_superset_api(query, query_params_json)
    
    # Output the response as JSON
    print(json.dumps(result))
