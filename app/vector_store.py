import chromadb
import hashlib
from app.embedder import get_embedding, chunk_text

# 初始化 ChromaDB，数据存在本地 chroma_db 目录
_client = chromadb.PersistentClient(path="./chroma_db")


def index_pdf(filename: str, pages: list[dict]) -> int:
    """
    把 PDF 内容切段、转向量、存入 ChromaDB。
    返回存入的段落数量。
    """
    # 每个 PDF 用文件名当作独立的 collection
    collection = _client.get_or_create_collection(name=_safe_name(filename))

    chunks = chunk_text(pages)

    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        collection.add(
            ids=[chunk["chunk_id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
        )

    return len(chunks)


def search_pdf(filename: str, question: str, top_k: int = 3) -> list[str]:
    """
    用问题在指定 PDF 里搜索最相关的段落。
    返回最相关的 top_k 段文字。
    """
    collection = _client.get_or_create_collection(name=_safe_name(filename))

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    return results["documents"][0]  # 返回最相关的段落列表

def _safe_name(filename: str) -> str:
    """
    ChromaDB collection 名称只接受英文/数字，
    用 hash 处理中文文件名。
    """
    # 取文件名的 md5 前 16 字，保证唯一且合法
    return "pdf_" + hashlib.md5(filename.encode()).hexdigest()[:16]