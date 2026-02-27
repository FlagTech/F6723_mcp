import os
import sys
import json
import asyncio
import time
import pickle
from typing import Callable
from dotenv import load_dotenv
from mcp import ClientSession
from google import genai
from rich.console import Console
from rich.markdown import Markdown
from mcp_utils import load_mcp, close_mcp

load_dotenv()
client = genai.Client()
console = Console()
afc_len = 0
hist_file = "chat_hist.pkl"

def set_afc_len(history:list[genai.types.Content]) -> int:
    global afc_len
    afc_len = len(history)

async def chat(
    sessions: list[ClientSession], 
    hooks: list[
        Callable[[genai.types.GenerateContentResponse], None]
    ]
):
    if os.path.exists(hist_file):
        console.print("接續對話")
        with open(hist_file, 'rb') as f:
            history = pickle.load(f)
            set_afc_len(history)
    else:
        history = None
    chat = client.aio.chats.create(
        model="gemini-2.5-flash",
        config=genai.types.GenerateContentConfig(
            tools=sessions,
            system_instruction=(
                f"現在 GMT 時間："
                f"{time.strftime("%c", time.gmtime())}\n"
                "請使用繁體中文"
                "以 Markdown 格式回覆"
            )
        ),
        history=history
    )
    
    while True:
        prompt = console.input("請輸入訊息(按 ⏎ 結束): ")  
        if prompt.strip() == "":
            break
        response = await chat.send_message(prompt)
        for hook in hooks:
            hook(response)
    
    history = chat.get_history()
    
    if history:
        with open(hist_file, 'wb') as f:
            pickle.dump(history, f)

def show_text(response: genai.types.GenerateContentResponse):
    console.print(Markdown(response.text))

def show_afc(response: genai.types.GenerateContentResponse):
    global afc_len
    if not response.automatic_function_calling_history:
        return
    for content in (
        response.automatic_function_calling_history[afc_len:]
    ):
        for part in content.parts:
            if part.function_call:
                name = part.function_call.name
                args = part.function_call.args
                console.print(f" →{name}(**{args})")
    afc_len = len(response.automatic_function_calling_history)

async def main():
    hooks = [show_afc, show_text]
    try:
        sessions = await load_mcp()
        await chat(sessions, hooks)
    except Exception as e:
        console.print(f"[red]錯誤: {e}[/red]")
    finally:
        await close_mcp()
        console.print("程式結束")

asyncio.run(main())
