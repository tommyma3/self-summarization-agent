"""Per-trial terminal execution. No policy or prompt history lives here."""
import asyncio
import base64
from concurrent.futures import CancelledError
from hashlib import sha256
import io
import json
from pathlib import Path
import shlex
import threading
import time
from uuid import uuid4


class TerminalBackend:
    def __init__(self, session, *, loop, logs_dir: Path, timeout: float = 600,
                 max_output_chars: int = 30000, vision=None):
        self.session = session
        self.loop = loop
        self.logs_dir = logs_dir
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.vision = vision
        self.cancelled = threading.Event()
        self._pending = None
        self._sequence = 0

    def cancel(self):
        self.cancelled.set()
        if self._pending is not None:
            self._pending.cancel()

    def execute(self, name, arguments, *, query_id):
        if self.cancelled.is_set():
            raise CancelledError("Trial cancelled")
        self._pending = asyncio.run_coroutine_threadsafe(self._execute(name, arguments), self.loop)
        try:
            return self._pending.result(timeout=self.timeout + 5)
        finally:
            self._pending.cancel()
            self._pending = None

    async def _execute(self, name, arguments):
        async with asyncio.timeout(self.timeout):
            if name == "execute_commands":
                output = await self._commands(arguments["commands"])
            elif name == "image_read" and self.vision is not None:
                output = await self.vision.read(self.session.environment, arguments, self.logs_dir)
            else:
                raise ValueError(f"Unavailable terminal tool: {name}")
        self._sequence += 1
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / f"observation-{self._sequence:05d}.txt").write_text(output)
        if len(output) > self.max_output_chars:
            return output[:self.max_output_chars] + "\n[Observation truncated; raw output saved in trial artifacts.]"
        return output

    async def _commands(self, commands):
        markers = set()
        for command in commands:
            keys = command["keystrokes"]
            duration = min(command.get("duration", 1.0), 60.0)
            if not keys:
                await asyncio.sleep(duration)
                continue
            await self.session.send_keys(keys, block=False, min_timeout_sec=0.0)
            marker = None
            if keys.endswith("\n"):
                marker = "__KIRA_DONE_" + uuid4().hex + "__"
                markers.add(marker)
                await self.session.send_keys(f"printf '\\n%s\\n' '{marker}'\n",
                                             block=False, min_timeout_sec=0.0)
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))
                if marker and marker in (await self.session.capture_pane()).splitlines():
                    break
        output = await self.session.get_incremental_output()
        return "\n".join(line for line in output.splitlines()
                         if not any(marker in line for marker in markers))


class FrozenVisionService:
    """Local frozen model; image bytes never leave the host for model inference."""

    def __init__(self, *, model_path: str, device: str = "cpu", max_tokens: int = 2048,
                 max_image_bytes: int = 10_000_000):
        if not Path(model_path).is_dir():
            raise ValueError("Vision model must be an existing local checkpoint directory")
        self.model_path = str(Path(model_path).resolve())
        self.device = device
        self.max_tokens = max_tokens
        self.max_image_bytes = max_image_bytes
        self._lock = threading.Lock()
        self._model = None

    def _analyze(self, data, instruction):
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
        with self._lock, torch.inference_mode():
            if self._model is None:
                self._processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path, local_files_only=True).to(self.device).eval()
                self._model.requires_grad_(False)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            inputs = self._processor.apply_chat_template([{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": instruction}
            ]}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            ids = self._model.generate(**inputs, do_sample=False, max_new_tokens=self.max_tokens)
            prompt_tokens = inputs["input_ids"].shape[-1]
            output_ids = ids[0, prompt_tokens:]
            return self._processor.decode(output_ids, skip_special_tokens=True), {
                "prompt_tokens": prompt_tokens, "completion_tokens": len(output_ids)}

    async def read(self, environment, arguments, logs_dir):
        path = arguments["file_path"]
        # Quote paths and bound the read before transferring data to the local host.
        result = await environment.exec(command=
            f"test -r {shlex.quote(path)} && head -c {self.max_image_bytes + 1} -- {shlex.quote(path)} | base64")
        if result.return_code:
            return "ERROR: Image read failed: " + (result.stderr or "")
        try:
            data = base64.b64decode("".join((result.stdout or "").split()), validate=True)
        except ValueError:
            return "ERROR: Invalid image transfer."
        if not data or len(data) > self.max_image_bytes:
            return "ERROR: Empty image or image exceeds configured size limit."
        output, usage = await asyncio.to_thread(self._analyze, data, arguments["image_read_instruction"])
        call = dict(model_path=self.model_path, image_sha256=sha256(data).hexdigest(),
                    file_path=path, instruction=arguments["image_read_instruction"],
                    output=output, usage=usage, trainable=False)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / f"vision-{uuid4().hex}.json").write_text(json.dumps(call))
        return f"Image analysis for {path}:\n{output}"
