import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from praxis.gateway import db
from praxis.gateway.router import Router
from praxis.gateway.routes import _openai_stream


class FakeStreamingProvider:
    name = "ollama"
    model = "fake-model"

    async def stream(self, messages, max_tokens=2048, temperature=0.7, model=None):
        yield "hello "
        yield "world"


class GatewayTelemetryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db_path = db.DB_PATH
        db.DB_PATH = str(Path(self.tmp.name) / "gateway.db")
        db.init()

    async def asyncTearDown(self):
        db.DB_PATH = self.original_db_path
        self.tmp.cleanup()

    async def test_stream_is_visible_while_running_then_completes(self):
        router = Router({"ollama": FakeStreamingProvider()}, ["ollama"])
        req = SimpleNamespace(max_tokens=20, temperature=0.2)
        stream = _openai_stream(
            router,
            [{"role": "user", "content": "Say hello"}],
            None,
            None,
            req,
        )

        first_chunk = await anext(stream)
        running = db.recent(limit=1)[0]
        self.assertIn("hello", first_chunk)
        self.assertEqual(running["status"], "running")

        remaining = [chunk async for chunk in stream]
        completed = db.recent(limit=1)[0]
        self.assertTrue(any("[DONE]" in chunk for chunk in remaining))
        self.assertEqual(completed["status"], "ok")
        self.assertGreater(completed["output_tokens"], 0)
        self.assertGreater(router.state["ollama"].tokens_today, 0)


if __name__ == "__main__":
    unittest.main()
