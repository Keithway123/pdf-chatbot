import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from dotenv import load_dotenv

from app.reader import extract_text_from_pdf
from app.store import save_pdf, get_pdf, list_pdfs
from app.vector_store import index_pdf, search_pdf

load_dotenv()

app = FastAPI(title="PDF Chatbot API", version="0.2.0")

# 初始化 LLM 客户端
llm_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)
LLM_MODEL = os.getenv("LLM_MODEL")


@app.get("/")
async def root():
    return {"message": "PDF Chatbot API 运行中"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只接受 PDF 文件")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pages = extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    # 存入内存
    save_pdf(file.filename, pages)

    # 切段、转向量、存入 ChromaDB
    chunk_count = index_pdf(file.filename, pages)

    return {
        "filename": file.filename,
        "total_pages": len(pages),
        "chunk_count": chunk_count,
    }


@app.get("/pdfs")
async def list_uploaded_pdfs():
    return {"pdfs": list_pdfs()}


@app.post("/ask")
async def ask(filename: str, question: str):
    # 检查 PDF 是否已上传
    if get_pdf(filename) is None:
        raise HTTPException(status_code=404, detail="找不到这个 PDF，请先上传")

    # 从向量数据库检索相关段落
    relevant_chunks = search_pdf(filename, question)

    # 把相关段落拼成 context
    context = "\n\n".join(relevant_chunks)

    # 组合 prompt 传给 LLM
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个文件问答助手，只根据提供的内容回答问题，不要编造。"
            },
            {
                "role": "user",
                "content": f"以下是文件内容：\n{context}\n\n问题：{question}"
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "filename": filename,
        "question": question,
        "answer": answer,
        "sources": relevant_chunks,
    }