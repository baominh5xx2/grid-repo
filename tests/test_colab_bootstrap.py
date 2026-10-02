"""Exercise notebook GitHub authentication without real credentials or network."""
import contextlib
import io
import json
import os
import pathlib
import subprocess
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sources():
    notebook = json.loads((ROOT / "notebooks/bf16_multiseed_main_method.ipynb").read_text(encoding="utf-8"))
    return {cell["metadata"]["tags"][0]: "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"}


class ColabBootstrapTests(unittest.TestCase):
    def auth_namespace(self):
        cells = sources()
        self.assertIn("github-auth", cells, "GitHub authentication must run before cloning")
        namespace = {}
        exec(cells["configuration"], namespace)
        return namespace, cells["github-auth"]

    def test_auth_runs_before_clone_and_installation(self):
        tags = list(sources())
        self.assertIn("github-auth", tags)
        self.assertLess(tags.index("github-auth"), tags.index("bootstrap"))
        self.assertLess(tags.index("bootstrap"), tags.index("installation"))

    def test_token_is_used_for_api_but_never_saved_or_printed(self):
        namespace, source = self.auth_namespace()
        token = "synthetic-secret-not-a-real-token"
        def response(request, timeout):
            self.assertEqual(request.full_url, "https://api.github.com/repos/baominh5xx2/grid-repo")
            self.assertEqual(request.get_header("Authorization"), "Bearer " + token)
            return contextlib.closing(io.BytesIO(json.dumps({"full_name": "baominh5xx2/grid-repo", "private": True}).encode()))
        output = io.StringIO()
        with patch.dict(os.environ, {"GITHUB_TOKEN": token}), patch("urllib.request.urlopen", side_effect=response), contextlib.redirect_stdout(output):
            exec(source, namespace)
            with namespace["github_git_environment"](namespace["_github_token"]) as env:
                helper = pathlib.Path(env["GIT_ASKPASS"])
                self.assertTrue(helper.is_file())
                self.assertNotIn(token, helper.read_text())
                self.assertEqual(env["GRID_GITHUB_TOKEN"], token)
                self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
                self.assertEqual(env["GIT_CONFIG_VALUE_0"], "")
                self.assertNotIn("GRID_GITHUB_TOKEN", os.environ)
            self.assertFalse(helper.exists())
        self.assertNotIn(token, output.getvalue())
        self.assertEqual(namespace["REPO_URL"], "https://github.com/baominh5xx2/grid-repo.git")

    def test_public_repo_can_clone_without_token(self):
        namespace, source = self.auth_namespace()
        response = contextlib.closing(io.BytesIO(json.dumps({"full_name": "baominh5xx2/grid-repo", "private": False}).encode()))
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", return_value=response), patch("getpass.getpass") as prompt:
            exec(source, namespace)
        prompt.assert_not_called()
        self.assertIsNone(namespace["_github_token"])

    def test_private_repo_prompts_and_checks_token(self):
        namespace, source = self.auth_namespace()
        response = contextlib.closing(io.BytesIO(json.dumps({"full_name": "baominh5xx2/grid-repo", "private": True}).encode()))
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", side_effect=[HTTPError("https://api.github.com", 404, "Not Found", {}, None), response]), patch("getpass.getpass", return_value="synthetic") as prompt:
            exec(source, namespace)
        prompt.assert_called_once()
        self.assertEqual(namespace["_github_token"], "synthetic")

    def test_bad_token_and_git_errors_do_not_echo_credentials(self):
        namespace, source = self.auth_namespace()
        with patch.dict(os.environ, {"GITHUB_TOKEN": "synthetic"}), patch("urllib.request.urlopen", side_effect=HTTPError("https://api.github.com", 401, "synthetic", {}, None)):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as failure:
                exec(source, namespace)
        self.assertNotIn("synthetic", str(failure.exception))
        completed = subprocess.CompletedProcess(["git"], 128, stdout="", stderr="synthetic")
        with patch("subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "Git failed") as failure:
                namespace["github_git"](["fetch", "origin", "main"], {})
        self.assertNotIn("synthetic", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
