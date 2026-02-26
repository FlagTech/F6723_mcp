import os
import sys
import json
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from contextlib import AsyncExitStack
from typing import Callable
from google import genai

async_exit_stack = AsyncExitStack()

async def get_remote_mcp_session(info:dict) -> ClientSession:
    if info.pop("type", None) == "http":
        read, write, _ = (
            await async_exit_stack.enter_async_context(
                streamable_http_client(**info)
            )
        )
    elif "url" in info:
        read, write = (
            await async_exit_stack.enter_async_context(
                sse_client(**info)
            )
        )
    elif "command" in info:
        stdio_server_params = StdioServerParameters(**info)
        read, write = (
            await async_exit_stack.enter_async_context(
                stdio_client(stdio_server_params)
            )
        )
    else:
        raise ValueError(f"未知的 MCP 伺服器類型: {info}")
    session = await async_exit_stack.enter_async_context(
        ClientSession(read, write)
    )
    await session.initialize()
    return session

async def load_mcp():
    sessions = []

    if (not os.path.exists("mcp_servers.json") or
        not os.path.isfile("mcp_servers.json")):
        return sessions

    with open('mcp_servers.json', 'r') as f:
        mcp_servers = json.load(f)
        try:
            server_infos = mcp_servers['mcp_servers'].items()
        except (KeyError, TypeError) as e:
            print(
                f"Error: mcp_servers.json 格式錯誤 - {e}",
                file=sys.stderr
            )
            return sessions

    for name, info in server_infos:
        print(f"啟動 MCP 伺服器 {name}...", end="")
        session = await get_remote_mcp_session(info)
        sessions.append(session)
        print(f"OK")
    return sessions

async def close_mcp():
    await async_exit_stack.aclose()

async def call_functions(
    response: genai.types.GenerateContentResponse,
    tools: list[Callable[[dict], str]],
    sessions: list[ClientSession],
    include_original_response: bool = True,
):
    results = []

    # 不需要叫用函式
    if not response.function_calls:
        return results
    
    # 先加入原本的回應
    if include_original_response:
        results.append(response.candidates[0].content)
    # 依序叫用函式
    for function_call in response.function_calls:
        name = function_call.name
        args = function_call.args
        result = None
        # 先檢查工具清單
        for tool in tools:
            if tool.__name__ == name:
                result = tool(**args)
                break
        # 如果沒有找到，再檢查 MCP 清單
        if result == None:
            for session in sessions:
                tool_list = await session.list_tools()
                for tool in tool_list.tools:
                    if tool.name == name:
                        result = (await session.call_tool(
                            name, 
                            args
                        )).content[0].text                    
                        break
                if not result == None:
                    break
        if not result == None:
            results.append(
                genai.types.Part.from_function_response(
                    name=name,
                    response={'result': result}
                )
            )
    return results
