from typing import Optional

# 简单的内存存储，之后阶段 3 会换成向量数据库
_storage: dict[str, list[dict]] = {}

# 存储对话历史，key 是 session_id
_chat_history: dict[str, list[dict]] = {}

def save_pdf(filename: str, pages: list[dict]) -> None:
    """储存解析好的 PDF 内容"""
    _storage[filename] = pages


def get_pdf(filename: str) -> Optional[list[dict]]:
    """取得已储存的 PDF 内容，找不到回传 None"""
    return _storage.get(filename)


def list_pdfs() -> list[str]:
    """列出所有已上传的 PDF 文件名"""
    return list(_storage.keys())

def get_history(session_id: str) -> list[dict]:
    """取得对话历史，没有则回传空列表"""
    return _chat_history.get(session_id, [])

def append_history(session_id: str, role: str, content: str) -> None:
    """新增一条消息到对话历史"""
    if session_id not in _chat_history:
        _chat_history[session_id] = []
    _chat_history[session_id].append({
        "role": role,
        "content": content
    })

def clear_history(session_id: str) -> None:
    """清除对话历史"""
    _chat_history.pop(session_id, None)