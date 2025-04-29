import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from dotenv import load_dotenv
import openai
import tiktoken
from typing import List

# Load OpenAI key
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Count tokens helper
def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Split text into chunks with overlap
def split_text(text, max_tokens=1500, overlap=200):
    words = text.split()
    chunks = []
    chunk = []
    token_count = 0

    for word in words:
        word_tokens = count_tokens(word)
        if token_count + word_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = chunk[-overlap:]
            token_count = count_tokens(" ".join(chunk))
        chunk.append(word)
        token_count += word_tokens

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

# Extract text from uploaded PDF
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# Summarize a chunk
def summarize_chunk(chunk):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize this document into bullet points."},
            {"role": "user", "content": chunk}
        ]
    )
    return response.choices[0].message.content.strip()

# Answer a question from a specific chunk
def answer_from_chunk(chunk, question):
    prompt = f"Based only on the following PDF chunk, answer the question.\n\nChunk:\n{chunk}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a PDF assistant that only uses provided information."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Store extracted content globally (in-memory, simple version)
pdf_chunks: List[str] = []

# Upload endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_chunks
    pdf_text = extract_text_from_pdf(file.file)
    pdf_chunks = split_text(pdf_text)

    return {"message": f"PDF uploaded and split into {len(pdf_chunks)} chunks."}

# Question endpoint
@app.post("/question")
async def ask_question(question: str = Form(...)):
    if not pdf_chunks:
        return JSONResponse(content={"error": "No PDF uploaded yet."}, status_code=400)

    # Find the best chunk based on keyword matches (simple search)
    best_chunk = pdf_chunks[0]
    best_score = 0

    for chunk in pdf_chunks:
        if question.lower().split()[0] in chunk.lower():
            score = chunk.lower().count(question.lower().split()[0])
            if score > best_score:
                best_score = score
                best_chunk = chunk

    answer = answer_from_chunk(best_chunk, question)
    return {"answer": answer}

# Summarize endpoint
@app.get("/summary")
async def summarize_pdf():
    if not pdf_chunks:
        return JSONResponse(content={"error": "No PDF uploaded yet."}, status_code=400)

    summaries = []
    for chunk in pdf_chunks:
        summaries.append(summarize_chunk(chunk))

    return {"summary": summaries}