from google import genai
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()
load_dotenv()
client = genai.Client()

response = client.models.generate_content_stream(
    model="gemini-3-pro-preview",
    contents="大谷翔平去年打了幾支全壘打？",
    config=genai.types.GenerateContentConfig(
        tools=[{"google_search": {}}],
        system_instruction="請用繁體中文回答"
    )
)

text = ""
# with Live(Markdown(""), console=console, refresh_per_second=10) as live:
#     for event in response:
#         text += event.text
#         live.update(Markdown(text))

live = Live(Markdown(""), console=console, refresh_per_second=10)
live.start()
for event in response:
    text += event.text
    live.update(Markdown(text))
live.stop()