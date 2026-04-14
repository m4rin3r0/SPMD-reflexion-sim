"""Browser-based launcher for the existing CLI simulation entry point."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import platform
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from threading import Timer
from urllib.parse import parse_qs
import webbrowser

from cli import build_parser


ROOT_DIR = Path(__file__).resolve().parent
CLI_PATH = ROOT_DIR / "cli.py"
PATH_FIELDS = {
    "json",
    "s2p",
    "s2p_zip",
    "jumped_s2p",
    "jumped_s2p_zip",
    "export_s2p",
    "export_jumped_s2p",
    "plot_repeated",
    "plot",
}
FIELD_ORDER = [
    ("Inputs", ["json", "s2p", "s2p_zip", "jumped_s2p", "jumped_s2p_zip"]),
    ("Frequency", ["freq_start", "freq_stop", "npoints", "z0"]),
    (
        "Topology",
        [
            "nodes",
            "length",
            "separation_min",
            "start_pad",
            "end_pad",
            "start_attach",
            "end_attach",
            "tx_node",
        ],
    ),
    ("Randomization", ["random_attach", "seed"]),
    ("Outputs", ["export_s2p", "export_jumped_s2p", "plot_repeated", "plot"]),
]


def _command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _terminal_command(shell_command: str) -> list[str]:
    system = platform.system()
    if system == "Darwin":
        script = (
            'tell application "Terminal"\n'
            "activate\n"
            f'do script "{_escape_applescript(shell_command)}"\n'
            "end tell"
        )
        return ["osascript", "-e", script]

    if system == "Windows":
        return ["cmd", "/k", shell_command]

    for terminal in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
        if not shutil.which(terminal):
            continue
        if terminal == "gnome-terminal":
            return [terminal, "--", "bash", "-lc", shell_command]
        if terminal == "konsole":
            return [terminal, "-e", "bash", "-lc", shell_command]
        if terminal == "xterm":
            return [terminal, "-hold", "-e", shell_command]
        return [terminal, "-e", "bash", "-lc", shell_command]

    raise RuntimeError("No supported terminal application was found.")


def _is_flag(action: object) -> bool:
    return getattr(action, "const", None) is True and getattr(action, "default", None) is False


def _collect_fields() -> dict[str, dict[str, object]]:
    parser = build_parser()
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest != "help"
    }
    fields: dict[str, dict[str, object]] = {}
    for _, names in FIELD_ORDER:
        for name in names:
            action = actions[name]
            fields[name] = {
                "name": name,
                "cli": action.option_strings[0],
                "help": action.help or "",
                "is_flag": _is_flag(action),
                "is_path": name in PATH_FIELDS,
            }
    return fields


FIELDS = _collect_fields()


def _build_command_from_values(values: dict[str, str]) -> list[str]:
    command = [sys.executable, str(CLI_PATH)]
    for _, names in FIELD_ORDER:
        for name in names:
            field = FIELDS[name]
            raw_value = values.get(name, "")
            if field["is_flag"]:
                if raw_value == "1":
                    command.append(str(field["cli"]))
                continue

            value = raw_value.strip()
            if not value:
                continue
            command.extend([str(field["cli"]), value])
    return command


def _launch_simulation(command: list[str]) -> None:
    shell_command = (
        f"cd {shlex.quote(str(ROOT_DIR))} && "
        f"{_command_to_string(command)}; "
        "status=$?; "
        'printf "\\nSimulation exited with status %s. Press Enter to close..." "$status"; '
        "read _"
    )
    subprocess.Popen(_terminal_command(shell_command), cwd=ROOT_DIR)


def _html_page(message: str = "", error: str = "", values: dict[str, str] | None = None) -> str:
    current = {name: "" for name in FIELDS}
    if values:
        current.update(values)

    sections: list[str] = []
    for title, names in FIELD_ORDER:
        rows: list[str] = []
        for name in names:
            field = FIELDS[name]
            value = current.get(name, "")
            label = html.escape(str(field["cli"]))
            help_text = html.escape(str(field["help"]))
            if field["is_flag"]:
                checked = " checked" if value == "1" else ""
                input_html = (
                    f'<label class="checkbox">'
                    f'<input type="checkbox" name="{name}" value="1"{checked}>'
                    f"<span>Enable</span>"
                    f"</label>"
                )
            else:
                placeholder = "Path" if field["is_path"] else "Value"
                input_html = (
                    f'<input type="text" name="{name}" value="{html.escape(value)}" '
                    f'placeholder="{placeholder}" autocomplete="off">'
                )

            rows.append(
                "<div class=\"field\">"
                f"<label>{label}</label>"
                f"{input_html}"
                f"<div class=\"help\">{help_text}</div>"
                "</div>"
            )
        sections.append(
            "<section>"
            f"<h2>{html.escape(title)}</h2>"
            f"{''.join(rows)}"
            "</section>"
        )

    command_preview = html.escape(_command_to_string(_build_command_from_values(current)))
    message_html = f'<div class="banner ok">{html.escape(message)}</div>' if message else ""
    error_html = f'<div class="banner err">{html.escape(error)}</div>' if error else ""
    schema = html.escape(json.dumps(FIELDS))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SPMD Reflection Simulator</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffdf8;
      --ink: #1f2a2e;
      --muted: #5d6a6f;
      --line: #d8cdbf;
      --accent: #8a4b2a;
      --accent-2: #224b5f;
      --ok: #e4f3e7;
      --ok-line: #8db092;
      --err: #fdeaea;
      --err-line: #ce8f8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(138, 75, 42, 0.12), transparent 24rem),
        radial-gradient(circle at bottom right, rgba(34, 75, 95, 0.12), transparent 26rem),
        var(--bg);
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 2rem 1rem 3rem;
    }}
    h1 {{
      margin: 0 0 0.5rem;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 1.5rem;
    }}
    form {{
      display: grid;
      gap: 1rem;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 1rem;
      box-shadow: 0 12px 40px rgba(31, 42, 46, 0.06);
    }}
    h2 {{
      margin: 0 0 0.8rem;
      font-size: 1.15rem;
      color: var(--accent-2);
    }}
    .field {{
      display: grid;
      grid-template-columns: 12rem 1fr;
      gap: 0.7rem 1rem;
      align-items: center;
      padding: 0.45rem 0;
      border-top: 1px solid rgba(216, 205, 191, 0.55);
    }}
    .field:first-of-type {{ border-top: 0; }}
    label {{
      font-family: "Courier New", monospace;
      font-size: 0.95rem;
    }}
    input[type="text"] {{
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 10px;
      padding: 0.65rem 0.8rem;
      font: inherit;
      color: var(--ink);
    }}
    .checkbox {{
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      font-family: inherit;
    }}
    .help {{
      grid-column: 2;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .preview {{
      background: #181c1d;
      color: #f0eee9;
      border-radius: 18px;
      padding: 1rem;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Courier New", monospace;
      line-height: 1.5;
      border: 1px solid #2d3538;
    }}
    .actions {{
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 0.8rem 1.2rem;
      font: inherit;
      cursor: pointer;
    }}
    .primary {{
      background: var(--accent);
      color: #fffaf6;
    }}
    .secondary {{
      background: var(--accent-2);
      color: #f8fcff;
    }}
    .banner {{
      border-radius: 14px;
      padding: 0.9rem 1rem;
      margin-bottom: 1rem;
    }}
    .ok {{
      background: var(--ok);
      border: 1px solid var(--ok-line);
    }}
    .err {{
      background: var(--err);
      border: 1px solid var(--err-line);
    }}
    @media (max-width: 760px) {{
      .field {{
        grid-template-columns: 1fr;
      }}
      .help {{
        grid-column: 1;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>SPMD Reflection Simulator</h1>
    <div class="sub">This GUI runs from the current Python environment and launches the existing CLI in a terminal.</div>
    {message_html}
    {error_html}
    <form method="post" action="/run" id="launcher-form">
      {''.join(sections)}
      <section>
        <h2>CLI Command</h2>
        <div class="preview" id="command-preview">{command_preview}</div>
      </section>
      <div class="actions">
        <button type="submit" class="primary">Run simulation</button>
        <button type="button" class="secondary" id="copy-command">Copy command</button>
      </div>
    </form>
  </main>
  <script id="field-schema" type="application/json">{schema}</script>
  <script>
    const form = document.getElementById("launcher-form");
    const preview = document.getElementById("command-preview");
    const schema = JSON.parse(document.getElementById("field-schema").textContent);
    function updatePreview() {{
      const data = new FormData(form);
      const command = [{json.dumps(sys.executable)}, {json.dumps(str(CLI_PATH))}];
      for (const [name, field] of Object.entries(schema)) {{
        if (field.is_flag) {{
          if (data.get(name) === "1") {{
            command.push(field.cli);
          }}
          continue;
        }}
        const value = (data.get(name) || "").toString().trim();
        if (!value) {{
          continue;
        }}
        command.push(field.cli, value);
      }}
      preview.textContent = command.map(part => {{
        if (/^[A-Za-z0-9_./:-]+$/.test(part)) {{
          return part;
        }}
        return JSON.stringify(part);
      }}).join(" ");
    }}
    form.addEventListener("input", updatePreview);
    document.getElementById("copy-command").addEventListener("click", async () => {{
      await navigator.clipboard.writeText(preview.textContent);
    }});
    updatePreview();
  </script>
</body>
</html>"""


class LauncherHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_html(_html_page())

    def do_POST(self) -> None:
        if self.path != "/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        parsed = parse_qs(body, keep_blank_values=True)
        values = {
            name: ("1" if parsed.get(name) == ["1"] else parsed.get(name, [""])[0])
            for name in FIELDS
        }

        try:
            command = _build_command_from_values(values)
            _launch_simulation(command)
        except Exception as exc:
            self._send_html(_html_page(error=str(exc), values=values), HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_html(
            _html_page(
                message="Simulation started in a terminal window.",
                values=values,
            )
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, content: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), LauncherHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(f"SPMD GUI listening on {url}")
    Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
