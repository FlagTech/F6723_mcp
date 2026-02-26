import asyncio
import time
from dotenv import load_dotenv
from mcp import ClientSession
from google import genai
from rich.console import Console
from rich.markdown import Markdown
from mcp_utils import load_mcp, close_mcp

load_dotenv()
client = genai.Client()
console = Console()

async def chat(sessions: list[ClientSession]):
    while True:
        prompt = console.input("請輸入訊息(按 ⏎ 結束): ")  
        if prompt.strip() == "":
            break
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                tools=sessions,
                system_instruction=(
                    f"現在 GMT 時間："
                    f"{time.strftime("%c", time.gmtime())}\n"
                    "請使用繁體中文"
                    "以 Markdown 格式回覆"
                ),
            ),
        )
        console.print(Markdown(response.text))

async def main():
    try:
        sessions = await load_mcp()
        await chat(sessions)
    except Exception as e:
        console.print(f"[red]錯誤: {e}[/red]")
    finally:
        console.print("程式結束")
        await close_mcp()

asyncio.run(main())