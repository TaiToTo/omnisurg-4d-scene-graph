#!/usr/bin/env python3
"""Refuse a git command that would put a personal email address in the repository.

Claude Code runs this before each Bash command. The command is refused when it
runs git and names an address that reaches a person, or sets an author or
committer whose address cannot be read from the command. The repository's
own `user.email`, the GitHub no-reply address, is the only identity a commit
takes. `tests/test_no_personal_email.py` reads the same rule.

Usage:
    echo '{"tool_name": "Bash", "tool_input": {"command": "git commit"}}' \\
        | python3 .claude/hooks/refuse_personal_email.py
"""

import json
import re
import sys

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Only addresses that reach no person are allowed.
#
# Match domains whole or as a parent domain, and addresses exactly, never as a
# substring: `"example.org" in addr` would also let through a real address
# that merely has the reserved name in the middle of its domain.
ALLOWED_DOMAINS = (
    # GitHub's no-reply addresses, one per account.
    "users.noreply.github.com",
    # Reserved by RFC 2606, together with every subdomain.
    "example.org",
    "example.com",
    "example.net",
)
ALLOWED_ADDRESSES = (
    # The committer of commits GitHub makes itself.
    "noreply@github.com",
    # The co-author trailer of commits made with Claude Code.
    "noreply@anthropic.com",
    # The user part of an SSH clone URL, `git@github.com:owner/repo.git`, which
    # a README is bound to show.
    "git@github.com",
)

# A git invocation: the word `git`, not a part of `digit` or `.gitignore`.
GIT = re.compile(r"(?<![\w.])git\b")
# A flag or variable that sets a commit's author or committer.
IDENTITY = re.compile(r"user\.email|--author\b|GIT_(?:AUTHOR|COMMITTER)_EMAIL")


def is_unreachable(addr: str) -> bool:
    """Whether `addr` reaches no person: an allowed address, or one under an allowed domain."""
    addr = addr.lower()
    if addr in ALLOWED_ADDRESSES:
        return True
    domain = addr.rpartition("@")[2]
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS)


def personal_emails(text: str) -> list[str]:
    """The addresses in `text` that may reach a person, sorted and without duplicates."""
    return sorted({m for m in EMAIL.findall(text) if not is_unreachable(m)})


def mask(addr: str) -> str:
    """`addr` with all but the first character of its local part hidden.

    A message that printed the address would publish it where the message
    goes: a CI log is public once the repository is.
    """
    local, _, domain = addr.partition("@")
    return f"{local[:1]}…@{domain}"


def refusal(tool_name: str, command: str) -> str | None:
    """Why `command` is refused, or None when it may run.

    A command that does not run git may run. One that runs git is refused
    when it names an address that is not allowed, or sets an identity
    without naming an address the hook can check.
    """
    if tool_name != "Bash" or not GIT.search(command):
        return None
    personal = personal_emails(command)
    if personal:
        return ("the command names an address that may reach a person: "
                + ", ".join(map(mask, personal))
                + ". A commit takes the repository's user.email, the GitHub no-reply address; "
                  "pass no address to git.")
    if IDENTITY.search(command) and not EMAIL.search(command):
        return ("the command sets an author or committer whose address it does not name, "
                "so the hook cannot check it. A commit takes the repository's user.email; "
                "pass no identity to git.")
    return None


def main() -> int:
    """Read the tool call from stdin, and exit 2 with the reason when the command is refused."""
    call = json.load(sys.stdin)
    command = call.get("tool_input", {}).get("command", "")
    why = refusal(call.get("tool_name", ""), command if isinstance(command, str) else "")
    if why is None:
        return 0
    print(f"refuse_personal_email: {why}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
