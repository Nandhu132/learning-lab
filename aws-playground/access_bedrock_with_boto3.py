import boto3
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def invoke_claude_3_5_sonnet(prompt):
    """
    Invokes Anthropic Claude 3.5 Sonnet via AWS Bedrock Runtime.
    """
    
    # Initialize the Bedrock Runtime client
    # Ensure you have configured AWS credentials and region
    try:
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1' 
        )
    except Exception as e:
        print(f"Error initializing Bedrock client: {e}")
        return

    # Model ID for Claude 3.5 Sonnet
    # Double check the model ID in your AWS Bedrock console if this fails
    model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

    # Prepare the payload for the Messages API
    # Claude 3.x models use the Messages API format
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps(payload)
        )

        # Parse the response
        response_body = json.loads(response.get('body').read())
        
        # Extract the text content
        # The response structure for Messages API is different from Text Completions
        if 'content' in response_body:
            for content_block in response_body['content']:
                if content_block['type'] == 'text':
                    print(content_block['text'])
        else:
             print("Unexpected response format:", response_body)

    except Exception as e:
        print(f"Error invoking model: {e}")

if __name__ == "__main__":
    user_prompt = "Hello, Claude! Tell me a fun fact about coding."
    print(f"Invoking Claude 3.5 Sonnet with prompt: '{user_prompt}'\n")
    invoke_claude_3_5_sonnet(user_prompt)
