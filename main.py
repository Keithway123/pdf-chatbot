import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
from openai.resources.skills import content

from app.reader import extract_text_from_pdf
from app.store import save_pdf, get_pdf, list_pdfs
from app.vector_store import index_pdf, search_pdf

from pydantic import BaseModel
from app.store import save_pdf, get_pdf, list_pdfs,get_history,append_history,clear_history

#统一错误处理
from fastapi.responses import JSONResponse
from fastapi.requests import Request

from fastapi.responses import StreamingResponse
import json

class AskRequest(BaseModel):
    filename: str
    question: str
    session_id: str ="default"

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
async def ask(request: AskRequest):
    # 检查 PDF 是否已上传
    if get_pdf(request.filename) is None:
        raise HTTPException(status_code=404, detail="找不到这个 PDF，请先上传")
    #1.取得历史对话
    history=get_history(request.session_id)
    # 2. 再用历史组合搜索查询
    recent_context = ""
    if history:
        last_messages = history[-2:] if len(history) >= 2 else history
        recent_context = " ".join([m["content"] for m in last_messages])

    search_query = f"{recent_context} {request.question}".strip()

    relevant_chunks = search_pdf(request.filename, search_query)
    context = "\n\n".join(relevant_chunks)


    #组合完整messages
    messages =[
        {
            "role":"system",
            "content":f"你是一个文件问答助手，只根据提供的内容回答问题，不要编造。\n\n文件内容：\n{context}"
        }
        ]
    messages.extend(history)
    messages.append(
        {
             "role":"user",
            "content":request.question
        }
    )

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages
    )
    answer = response.choices[0].message.content

    #把这轮对话存入历史
    append_history(request.session_id, "user", request.question)
    append_history(request.session_id, "assistant", answer)

    return {
        "filename": request.filename,
        "question": request.question,
        "answer": answer,
        "session_id":request.session_id,
        "source":relevant_chunks,
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id:str):
    clear_history(session_id)
    return {"message":f"session {session_id} cleared"}

@app.exception_handler(Exception)
async def global_exception_handle(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误",
            "detail": str(exc)
        }
    )


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    if get_pdf(request.filename) is None:
        raise HTTPException(status_code=404, detail="找不到这个 PDF，请先上传")

    history = get_history(request.session_id)

    recent_context = ""
    if history:
        last_messages = history[-2:] if len(history) >= 2 else history
        recent_context = " ".join([m["content"] for m in last_messages])

    search_query = f"{recent_context} {request.question}".strip()
    relevant_chunks = search_pdf(request.filename, search_query)
    context = "\n\n".join(relevant_chunks)

    messages = [
        {
            "role": "system",
            "content": f"你是一个文件问答助手，只根据提供的内容回答问题。\n\n文件内容：\n{context}"
        }
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": request.question})

    # 这里需要你写：
    # 1. 定义一个 generator 函数 generate()
    # 2. 呼叫 LLM 时加上 stream=True
    # 3. 每收到一个 chunk 就 yield 出去
    # 4. 最后 yield "[DONE]"
    # 5. 用 StreamingResponse 返回

    async def generate():
        stream = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                yield f"data:{text}\n\n"
        yield "data:[DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
