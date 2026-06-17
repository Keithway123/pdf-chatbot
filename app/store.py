from typing import Optional

# 简单的内存存储，之后阶段 3 会换成向量数据库
_storage: dict[str, list[dict]] = {}


def save_pdf(filename: str, pages: list[dict]) -> None:
    """储存解析好的 PDF 内容"""
    _storage[filename] = pages


def get_pdf(filename: str) -> Optional[list[dict]]:
    """取得已储存的 PDF 内容，找不到回传 None"""
    return _storage.get(filename)


def list_pdfs() -> list[str]:
    """列出所有已上传的 PDF 文件名"""
    return list(_storage.keys())