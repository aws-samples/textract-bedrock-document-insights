import os
import time
import json
from typing import Optional
import boto3
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import io
from PyPDF2 import PdfReader
import tempfile

# Load environment variables
load_dotenv()

# Define environment variables or default values
S3_BUCKET = os.environ.get("S3_BUCKET")
if not S3_BUCKET:
    st.error("S3_BUCKET environment variable is not set!")
    
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = "amazon.nova-micro-v1:0"

def upload_to_s3(file_obj, bucket, key):
    """Upload a file to S3"""
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3_client.upload_fileobj(file_obj, bucket, key)
        return True
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return False

def invoke_bedrock_model(client: boto3.client, prompt: str, extracted_text: str) -> Optional[str]:
    system_list = [
        {
            "text": "You are a helpful assistant that analyzes text from scanned documents"
        }
    ]

    message_list = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"{prompt}:\n\n{extracted_text}\n\n"
                }
            ]
        }
    ]

    inf_params = {
        "max_new_tokens": 1000,
        "top_p": 0.9,
        "top_k": 20,
        "temperature": 0.7
    }

    request_body = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params
    }

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response['body'].read())
        
        if "output" in response_body:
            message = response_body["output"]["message"]
            if "content" in message and len(message["content"]) > 0:
                return message["content"][0]["text"]
        return ""
    
    except Exception as e:
        st.error(f"Error invoking model: {str(e)}")
        return ""

def process_document(s3_key, custom_prompt):
    """
    Process document with Textract and Bedrock.
    
    Args:
        s3_key (str): S3 object key of the uploaded document
        custom_prompt (str): Custom prompt for Bedrock analysis
    
    Returns:
        dict: Dictionary containing:
            - extracted_text: Text extracted from document
            - analysis_result: Analysis from Bedrock
            - textract_time: Time taken by Textract
            - bedrock_time: Time taken by Bedrock
    """
    try:
        # Initialize AWS clients
        textract_client = boto3.client("textract", region_name=AWS_REGION)
        bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

        document = {
            "S3Object": {
                "Bucket": S3_BUCKET,
                "Name": s3_key,
            }
        }

        # Process with Textract and measure time
        textract_start = time.time()
        with st.spinner('Processing document with Textract...'):
            detect_text_output = textract_client.detect_document_text(Document=document)
            extracted_text = "\n".join(
                [block["Text"] for block in detect_text_output["Blocks"] if "Text" in block]
            )
        textract_time = time.time() - textract_start

        # Process with Bedrock and measure time
        bedrock_start = time.time()
        with st.spinner('Analyzing with Bedrock...'):
            analysis_result = invoke_bedrock_model(bedrock_client, custom_prompt, extracted_text)
        bedrock_time = time.time() - bedrock_start
            
        return {
            "extracted_text": extracted_text,
            "analysis_result": analysis_result,
            "textract_time": textract_time,
            "bedrock_time": bedrock_time
        }
    
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return {
            "extracted_text": "",
            "analysis_result": "",
            "textract_time": 0,
            "bedrock_time": 0
        }

def main():
    st.set_page_config(page_title="Document Analysis with AWS", layout="wide")
    
    st.title("Low latency document Analysis with AWS")
    st.write("Upload a document and analyze it using AWS Textract and Bedrock- Nova Micro")

    # Main content area
    col1, col2 = st.columns([1, 1])

    # Add a sidebar for inference parameters
    with st.sidebar:
        st.header("Inference Parameters")
        max_new_tokens = st.slider(
            label="Maximum number of tokens to generate",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100
        )
        temperature = st.slider(
            label="Temperature (controls randomness)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        top_p = st.slider(
            label="Top P (nucleus sampling)",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1
        )
        top_k = st.slider(
            label="Top K (number of tokens to consider)",
            min_value=1,
            max_value=100,
            value=20,
            step=1
        )
        
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            label="Upload your document",
            type=['png', 'jpg', 'jpeg', 'pdf']
        )
        
        # Custom prompt input
        default_prompt = "Extract the following details from chemistry lab notes into CSV format: Chemical Compound Name, Initial Temperature (°C), Final Temperature (°C), Reaction Time (min). If any value is not specified, leave it blank. Output only the CSV record."
        custom_prompt = st.text_area(
            label="Enter your analysis prompt",
            value=default_prompt,
            height=200,
            help="Specify how you want the document to be analyzed"
        )

        # Preview handling for different file types
        if uploaded_file is not None:
            file_type = uploaded_file.type
            if file_type == "application/pdf":
                try:
                    # Create a temporary file to store the PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file.flush()  # Flush the file buffer
                        tmp_file_path = tmp_file.name

                    # Read PDF and check number of pages
                    pdf_reader = PdfReader(tmp_file_path)
                    num_pages = len(pdf_reader.pages)
                    
                    if num_pages > 1:
                        st.error("Multi-page documents are not supported for this demonstration. Please upload a single-page document.")
                        uploaded_file = None
                    else:
                        st.write("PDF document preview:")
                        # Display the text content of the page
                        page = pdf_reader.pages[0]
                        st.text_area(
                            label="PDF content",
                            value=page.extract_text(),
                            height=300,
                            disabled=True
                        )
                    
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    uploaded_file = None
                    
            elif file_type.startswith('image/'):
                st.write("Image preview:")
                st.image(uploaded_file, caption="Preview of uploaded document", use_container_width=True)

    with col2:
        if uploaded_file and st.button("Process Document", type="primary"):
            total_start = time.time()
            
            file_extension = uploaded_file.name.split('.')[-1]
            s3_key = f"uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"

            with st.spinner('Uploading file to S3...'):
                if upload_to_s3(uploaded_file, S3_BUCKET, s3_key):
                    st.success("File uploaded successfully!")
                    
                    # Get results as a dictionary
                    result = process_document(s3_key, custom_prompt)
                    total_time = time.time() - total_start
                    
                    # Display metrics
                    col1_metric, col2_metric, col3_metric = st.columns(3)
                    with col1_metric:
                        st.metric(label="Textract Processing Time", value=f"{result['textract_time']:.2f}s")
                    with col2_metric:
                        st.metric(label="Bedrock Analysis Time", value=f"{result['bedrock_time']:.2f}s")
                    with col3_metric:
                        st.metric(label="Total Processing Time", value=f"{total_time:.2f}s")
                    
                    # Display results
                    st.subheader("Extracted Text")
                    st.text_area(
                        label="Text extracted from document",
                        value=result['extracted_text'],
                        height=200,
                        key="extracted_text"
                    )
                    
                    st.subheader("Analysis Result")
                    st.text_area(
                        label="AI analysis results",
                        value=result['analysis_result'],
                        height=200,
                        key="analysis_result"
                    )
                else:
                    st.error("Failed to upload file")

if __name__ == "__main__":
    main()
