from pathlib import Path
import fitz  # pymupdf


def extract_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    """
    读取 PDF，返回每一页的文字内容。

    返回格式：
    [
        {"page": 1, "text": "第一页内容..."},
        {"page": 2, "text": "第二页内容..."},
    ]
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到文件：{pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"不是 PDF 文件：{pdf_path}")

    pages = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            pages.append({
                "page": page_num,
                "text": text.strip()
            })

    return pages


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python -m app.reader <pdf文件路径>")
        sys.exit(1)

    results = extract_text_from_pdf(sys.argv[1])

    for item in results:
        print(f"\n{'='*40}")
        print(f"第 {item['page']} 页")
        print(f"{'='*40}")
        print(item["text"][:500])  # 只印前500字，避免刷屏