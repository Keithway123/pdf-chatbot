import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化客户端，指向阿里云百炼
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")


def get_embedding(text: str) -> list[float]:
    """
    把一段文字转成向量。
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def chunk_text(pages: list[dict], chunk_size: int = 500) -> list[dict]:
    """
    把 PDF 的页面内容切成小段。

    返回格式：
    [
        {"chunk_id": "page1_0", "text": "第一段内容..."},
        {"chunk_id": "page1_1", "text": "第二段内容..."},
    ]
    """
    chunks = []
    for page in pages:
        text = page["text"]
        page_num = page["page"]

        # 每 chunk_size 个字切一段
        for i in range(0, len(text), chunk_size):
            chunk = text[i: i + chunk_size]
            if chunk.strip():  # 跳过空白段
                chunks.append({
                    "chunk_id": f"page{page_num}_{i}",
                    "text": chunk
                })

    return chunks