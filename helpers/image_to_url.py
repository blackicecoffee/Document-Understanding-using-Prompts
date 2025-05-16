import base64

# Function to encode a local image into data URL 
def image_to_data_url(image_path: str) -> str:
    # Guess the MIME type of the image based on the file extension
    mime_type = "image/png"

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"