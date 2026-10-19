"""The hook refuses a git command that names a personal address or hides an identity.

`.claude/hooks/refuse_personal_email.py` runs before each Bash command under
Claude Code. Each refusal has a test that plants its fault in a command; a
command that runs no git, or names only addresses that reach no person, runs.
The hook is run as Claude Code runs it, as a process reading JSON on stdin,
and `.claude/settings.json` has to name it.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / ".claude" / "hooks" / "refuse_personal_email.py"


def _addr(local, domain):
    """Put a sample address together, so that it never appears as a literal."""
    return local + "@" + domain


NOREPLY = _addr("1+someone", "users.noreply.github.com")
PERSONAL = _addr("first.last.ml", "gmail.com")


def run(command: str, tool_name: str = "Bash") -> subprocess.CompletedProcess:
    call = {"tool_name": tool_name, "tool_input": {"command": command}}
    return subprocess.run([sys.executable, str(HOOK)], input=json.dumps(call).encode(),
                          capture_output=True)


@pytest.mark.parametrize("command", [
    "git commit -q -m 'a message'",
    "cd /somewhere && git commit --amend --no-edit --reset-author",
    f"git config user.email {NOREPLY}",
    f"git -c user.email={NOREPLY} commit -m x",
    "git commit -m 'x' -m 'Co-Authored-By: Claude <noreply@anthropic.com>'",
    f"echo {PERSONAL}",
    "ls .gitignore digit",
])
def test_a_command_without_a_personal_address_or_a_hidden_identity_runs(command):
    done = run(command)
    assert done.returncode == 0 and done.stderr == b""


@pytest.mark.parametrize("command", [
    f"git -c user.email={PERSONAL} commit -m x",
    f"git config user.email {PERSONAL}",
    f"cd /x && git commit --author='A <{PERSONAL}>' -m x",
    f"GIT_COMMITTER_EMAIL={PERSONAL} git commit -m x",
    f"git commit -m 'Co-Authored-By: Someone <{PERSONAL}>'",
    f"git tag -a v1 -m 'contact {PERSONAL}'",
])
def test_a_git_command_naming_a_personal_address_is_refused_with_the_address_masked(command):
    done = run(command)
    assert done.returncode == 2
    assert b"may reach a person" in done.stderr
    assert PERSONAL.encode() not in done.stderr and b"@gmail.com" in done.stderr


@pytest.mark.parametrize("command", [
    "GIT_AUTHOR_EMAIL=$EMAIL git commit -m x",
    "git -c user.email=\"$(cat ~/.email)\" commit -m x",
    "git commit --author=\"$AUTHOR\" -m x",
])
def test_a_git_command_setting_an_identity_without_an_address_is_refused(command):
    done = run(command)
    assert done.returncode == 2 and b"does not name" in done.stderr


def test_another_tool_is_not_judged():
    assert run(f"git -c user.email={PERSONAL} commit", tool_name="Read").returncode == 0


def test_the_settings_run_the_hook_before_each_bash_command():
    settings = json.loads((REPO / ".claude" / "settings.json").read_text())
    [entry] = [e for e in settings["hooks"]["PreToolUse"] if e["matcher"] == "Bash"]
    [hook] = entry["hooks"]
    assert hook["type"] == "command" and hook["onFailure"] == "block"
    assert hook["command"].endswith(HOOK.relative_to(REPO).as_posix())


def test_the_hook_and_its_settings_are_tracked():
    tracked = subprocess.run(["git", "ls-files", ".claude"], cwd=REPO, check=True,
                             capture_output=True).stdout.decode().split()
    assert set(tracked) >= {".claude/settings.json", ".claude/hooks/refuse_personal_email.py"}
