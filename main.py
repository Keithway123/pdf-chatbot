from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os

from app.reader import extract_text_from_pdf

app = FastAPI(title="PDF Chatbot API", version="0.1.0")


@app.get("/")
async def root():
    return {"message": "PDF Chatbot API 运行中"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 检查是否为 PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只接受 PDF 文件")

    # 把上传的文件暂存到本地，再交给 reader 处理
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pages = extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)  # 用完删掉暂存文件

    return {
        "filename": file.filename,
        "total_pages": len(pages),
        "pages": pages
    }