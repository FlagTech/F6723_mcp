from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

def google_search(query: str) -> str:
    """
    Google 搜尋。

    Args:
        query (str): 要查詢的關鍵字

    Returns:
        str: 以 Markdown 條列格式返回搜尋結果
    """

    # 範例使用：讓模型呼叫這個自訂搜尋工具
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        config={
            "tools": [{"google_search": {}}],
            "system_instruction": "請用繁體中文回答"
        },
        contents=(
            f"搜尋以下關鍵字後整理成 Markdown 條列格式：\n"
            f"{query}"
        )
    )
    return response.text
