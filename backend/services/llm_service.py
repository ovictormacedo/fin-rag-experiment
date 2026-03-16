import json
from urllib.error import URLError
from urllib.request import Request, urlopen

class LLMService:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model = "gemma3:4b"

    def post(self, prompt):
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = Request(f"{self.ollama_url}/api/generate", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "").strip()
        except URLError as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e