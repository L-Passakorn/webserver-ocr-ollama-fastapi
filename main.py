from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import ollama
import io

# Initialize FastAPI app
app = FastAPI()

# Define allowed origins (for development, you can allow localhost, but in production, limit it)
origins = [
    "http://localhost:3000",  # Your React app's URL (local)
    "http://127.0.0.1:3000",  # Optional: another way localhost might be accessed
    # Add other domains that need to access the FastAPI server
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Domains that are allowed to access the API
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Specify Tesseract-OCR executable path (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Endpoint for uploading the image and getting summarized text


@app.post("/summarize-ocr/")
async def summarize_ocr(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Image processing steps to improve OCR
    grayscale_image = image.convert('L')
    threshold_image = grayscale_image.point(lambda x: 0 if x < 140 else 255)
    resized_image = threshold_image.resize(
        (threshold_image.width * 2, threshold_image.height * 2), Image.Resampling.LANCZOS)
    denoised_image = resized_image.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(denoised_image)
    enhanced_image = enhancer.enhance(2)

    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string(enhanced_image, lang="eng")

    if not extracted_text:
        return {"error": "No text could be extracted from the image."}

    # Send extracted text to the Ollama model with clear instructions
    summary_response = ollama.chat(
        model="supachai/llama-3-typhoon-v1.5",
        messages=[
            {
                "role": "user",
                "content": (
                    # f"Please summarize the following text using a structured format:\n"
                    # f"* Each main point should be a header, started with an asterisk (*).\n"
                    # f"* Details should be started with double asterisks (**).\n"
                    # f"* Further sub-details should be started with triple asterisks (***).\n"
                    # f"* Do not end the line with any amount of asterisk.\n"
                    # f"* Please make sure it is clear and concise.\n"
                    # f"{extracted_text}"

                    # f"Please summarize the following text using a structured format:\n\n"
                    # f"@startuml\n"
                    # f"skinparam shadowing false\n"
                    # f"actor Alice\n"
                    # f"actor Bob\n"
                    # f"Alice -> Bob: Authentication Request\n"
                    # f"Bob --> Alice: Authentication Response\n"
                    # f"Alice -> Bob: Another authentication Request\n"
                    # f"Alice <-- Bob: Another authentication Response\n"
                    # f"@enduml\n\n"

                    f"Please summarize the following text into a Mermaid diagram in the graph TD format.\n\n"
                    f"Make sure to use nodes and directional edges with the following structure:\n\n"
                    f"graph TD;\n"
                    f"  Title-->Header1;\n"
                    f"  Header1-->Detail;\n"
                    f"  Header1-->Detail;\n"
                    f"  Header1-->Detail;\n"
                    f"  Title-->Header2;\n"
                    f"  Header2-->Detail;\n"
                    f"  Title-->Header3;\n"
                    f"  Header3-->Detail;\n"
                    f"  Header3-->Detail;\n"
                    f"* Title Header1 and Detail1 just for the example\n"
                    f"* The number of header and detail can be different from the example\n"
                    f"* Please make sure it is a valid Mermaid diagram\n"
                    f"Here is the text to be summarized:\n"
                    f"{extracted_text}"
                )
            }
        ]
    )

    # Check for 'content' inside the 'message' object
    if "message" in summary_response and "content" in summary_response["message"]:
        summary = summary_response["message"]["content"]
    else:
        # Log the entire response if 'content' is missing for debugging
        return {"error": "Unexpected response from Ollama", "response": summary_response}

    # Return the extracted text and summary
    return {"extracted_text": extracted_text, "summary": summary}
