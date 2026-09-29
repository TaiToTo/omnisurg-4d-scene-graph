"""No personal email address reaches the repository.

Names and affiliations may appear: they are printed on the paper. Email
addresses may not. Once pushed to a public repository they are indexed and
harvested, and taking one out afterwards means rewriting history.

An address gets in by three routes, and this checks all of them:

- **git metadata.** The author and committer of every commit, and the tagger
  of every annotated tag. A development machine's global git config usually
  holds a personal address, and a commit GitHub makes itself (a merge or an
  edit in the web UI) carries whichever address the account exposes.
- **commit messages.** Trailers such as `Co-authored-by:`, which GitHub fills
  in from the co-author's account when a pull request is squashed.
- **tracked file contents.**

The history checked is HEAD's, plus every tag. In CI on a pull request, HEAD
is the pull request merged into main, so every commit is checked before it
reaches main. The one exception is the merge commit GitHub makes when the pull
request is merged; the run on the push to main checks that one.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

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
)

# Files whose bytes are not text. Compressed image data produces strings that
# look like addresses, so these are not scanned. A PDF is deliberately not
# here: it can carry a contact address in its text, which a byte scan cannot
# see, so a tracked PDF fails as unreadable instead of passing unseen.
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf"}

# A field separator for `git log --format`: a name, an address or a message
# cannot contain it.
_FS = "\x1f"


def _git(*args, cwd=REPO):
    """Run git and return its standard output.

    Args:
        *args: The git subcommand and its arguments.
        cwd: The repository to run in.

    Returns:
        The output, decoded as UTF-8.

    Raises:
        subprocess.CalledProcessError: git failed, for one because `cwd` is
            not a repository. The check cannot be made, so it is not passed.
    """
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True).stdout.decode("utf-8")


def is_unreachable(addr):
    """Tell whether an address is known to reach no person.

    Args:
        addr: One address.

    Returns:
        True if the allowlist covers it.
    """
    addr = addr.lower()
    if addr in ALLOWED_ADDRESSES:
        return True
    domain = addr.rpartition("@")[2]
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS)


def personal_emails(text):
    """List the addresses in a text that may reach a person.

    Args:
        text: The text to scan.

    Returns:
        The addresses found, sorted and without duplicates.
    """
    return sorted({m for m in EMAIL.findall(text) if not is_unreachable(m)})


def mask(addr):
    """Hide the local part of an address, for failure messages.

    CI logs are public once the repository is. A failure that printed the
    address it found would publish it there.

    Args:
        addr: One address.

    Returns:
        The address with all but the first character of its local part hidden.
    """
    local, _, domain = addr.partition("@")
    return f"{local[:1]}…@{domain}"


def history(root=REPO):
    """List every place in the git history where an address can sit.

    Args:
        root: The repository.

    Returns:
        A list of (object, field, text) triples. `object` is a commit or tag.
        `field` is "author", "committer", "tagger" or "message". `text` is the
        address for the first three and the whole message for the last.

    Raises:
        RuntimeError: The clone is shallow. The commits it lacks cannot be
            checked, so the history cannot be called clean.
    """
    if _git("rev-parse", "--is-shallow-repository", cwd=root).strip() == "true":
        raise RuntimeError(
            "the clone is shallow, so part of the history is missing and cannot "
            "be checked. Fetch it whole (actions/checkout: fetch-depth: 0).")
    out = []
    log = _git("log", "-z", f"--format=%H{_FS}%ae{_FS}%ce{_FS}%B", "HEAD", "--tags", cwd=root)
    for record in filter(None, log.split("\0")):
        sha, author, committer, message = record.split(_FS, 3)
        out += [(sha, "author", author), (sha, "committer", committer),
                (sha, "message", message)]
    tags = _git("for-each-ref", "refs/tags",
                f"--format=%(objecttype){_FS}%(refname:short){_FS}%(taggeremail){_FS}%(contents)%00",
                cwd=root)
    for record in filter(None, (r.lstrip("\n") for r in tags.split("\0"))):
        kind, name, tagger, message = record.split(_FS, 3)
        if kind == "tag":                          # a lightweight tag has no tagger
            out += [(name, "tagger", tagger.strip("<>")), (name, "message", message)]
    return out


def personal_emails_in_history(root=REPO):
    """Find the addresses in the history that may reach a person.

    An author, committer or tagger address must be on the allowlist: one
    without a top-level domain (`someone@laptop`) names a person and a machine
    as surely as a real one. A message is scanned for addresses.

    Args:
        root: The repository.

    Returns:
        A sorted list of (object, field, masked address) triples.
    """
    found = set()
    for obj, field, text in history(root):
        if field == "message":
            found |= {(obj[:12], field, mask(a)) for a in personal_emails(text)}
        elif not is_unreachable(text):
            found.add((obj[:12], field, mask(text)))
    return sorted(found)


def tracked_files(root=REPO):
    """List the tracked files.

    Read with `-z`: without it, git quotes a name that is not ASCII, the quoted
    name opens nothing, and that file drops out of the scan without a word.

    Args:
        root: The repository.

    Returns:
        Paths relative to `root`.
    """
    return [f for f in _git("ls-files", "-z", cwd=root).split("\0") if f]


def tracked_text_files(root=REPO):
    """List the tracked files that are scanned as text.

    Args:
        root: The repository.

    Returns:
        Paths relative to `root`.
    """
    return [f for f in tracked_files(root) if Path(f).suffix.lower() not in BINARY_SUFFIXES]


def read_text_strictly(path):
    """Read a file as text, or raise.

    Reading with `errors="ignore"` would drop what does not decode, and pass a
    file whose address sits in the dropped part (an address in UTF-16 is full
    of NULs and never matches `EMAIL`).

    Args:
        path: The file.

    Returns:
        Its contents.

    Raises:
        ValueError: It is not UTF-8, or it contains a NUL.
    """
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ValueError("contains a NUL: UTF-16 or binary")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"not UTF-8: {exc}") from exc


def personal_emails_in_files(root=REPO):
    """Find the addresses in tracked files that may reach a person.

    Args:
        root: The repository.

    Returns:
        A pair of dicts: path to masked addresses found, and path to why the
        file could not be read. A file that cannot be read has not been
        checked, so it is reported rather than skipped.
    """
    found, unreadable = {}, {}
    for rel in tracked_text_files(root):
        path = Path(root) / rel
        if not path.is_file():
            unreadable[rel] = "tracked, but not a file that can be opened"
            continue
        try:
            hits = personal_emails(read_text_strictly(path))
        except ValueError as exc:
            unreadable[rel] = str(exc)
            continue
        if hits:
            found[rel] = [mask(a) for a in hits]
    return found, unreadable


def test_no_commit_or_tag_carries_a_personal_email():
    found = personal_emails_in_history()
    assert not found, (
        "the git history carries an address that may reach a person. Rewrite "
        "those commits before they are pushed, and set this repository's "
        "user.email to the GitHub no-reply address:\n"
        + "\n".join(f"  {obj} {field}: {addr}" for obj, field, addr in found))


def test_no_tracked_file_carries_a_personal_email():
    found, unreadable = personal_emails_in_files()
    assert not unreadable, (
        "tracked files that could not be scanned as text, so they have not been "
        "checked. Add a binary format to BINARY_SUFFIXES, or do not track it:\n"
        + "\n".join(f"  {rel}: {why}" for rel, why in sorted(unreadable.items())))
    assert not found, (
        "tracked files carry an address that may reach a person:\n"
        + "\n".join(f"  {rel}: {hits}" for rel, hits in sorted(found.items())))


def test_the_history_scan_sees_every_commit():
    # A parsing mistake (a separator split wrongly) would drop commits from the
    # scan, and the check above would pass on what it never saw.
    commits = {obj for obj, field, _ in history() if field == "author"}
    expected = _git("rev-list", "HEAD", "--tags").split()
    assert commits == set(expected)


# Share of tracked files scanned as text. A ratio, not a count, so that it
# holds while the repository grows.
MIN_TEXT_RATIO = 0.9


def test_the_file_scan_actually_looks_at_files():
    # Widening BINARY_SUFFIXES too far would shrink the scan until it passes
    # on nothing.
    files = tracked_files()
    assert files, "no tracked file was listed"
    assert len(tracked_text_files()) / len(files) >= MIN_TEXT_RATIO


# ------------------------------------------------------ when it should fail
#
# No sample address is written out in this file: the file is tracked, so the
# scan above would find it and the file would have to exempt itself. Samples
# are put together from their two halves instead.


def _addr(local, domain):
    """Put a sample address together, so that it never appears as a literal."""
    return local + "@" + domain


NOREPLY = _addr("1+someone", "users.noreply.github.com")
PERSONAL = _addr("first.last.ml", "gmail.com")


def _env(author=NOREPLY, committer=NOREPLY):
    """An environment for git that ignores this machine's git config.

    The global config could set a signing key or an identity, and the tests
    would then check this machine rather than the scanner.
    """
    return {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "A", "GIT_AUTHOR_EMAIL": author,
            "GIT_COMMITTER_NAME": "C", "GIT_COMMITTER_EMAIL": committer}


def _repo(root, commits):
    """Make a repository with one empty commit per entry.

    Args:
        root: The directory to make it in.
        commits: (author, committer, message) triples.

    Returns:
        `root`.
    """
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True, env=_env())
    for author, committer, message in commits:
        subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", message],
                       cwd=root, check=True, env=_env(author, committer))
    return root


def test_a_clean_history_passes(tmp_path):
    # Without this, the tests below would also pass on a scanner that flags
    # everything.
    root = _repo(tmp_path, [(NOREPLY, NOREPLY, "first"), (NOREPLY, NOREPLY, "second")])
    assert personal_emails_in_history(root) == []


def test_a_personal_author_is_caught(tmp_path):
    root = _repo(tmp_path, [(NOREPLY, NOREPLY, "first"), (PERSONAL, NOREPLY, "second")])
    assert [f for _, f, _ in personal_emails_in_history(root)] == ["author"]


def test_a_personal_committer_is_caught(tmp_path):
    # A rebase or a cherry-pick keeps the author and rewrites the committer.
    root = _repo(tmp_path, [(NOREPLY, PERSONAL, "first")])
    assert [f for _, f, _ in personal_emails_in_history(root)] == ["committer"]


def test_a_personal_co_author_trailer_is_caught(tmp_path):
    root = _repo(tmp_path, [(NOREPLY, NOREPLY, f"squash\n\nCo-authored-by: B <{PERSONAL}>")])
    assert [f for _, f, _ in personal_emails_in_history(root)] == ["message"]


def test_an_author_address_without_a_domain_is_caught(tmp_path):
    # What git falls back to when no user.email is set: the login and the
    # machine's name.
    root = _repo(tmp_path, [(_addr("someone", "laptop"), NOREPLY, "first")])
    assert [f for _, f, _ in personal_emails_in_history(root)] == ["author"]


def test_a_personal_tagger_is_caught(tmp_path):
    root = _repo(tmp_path, [(NOREPLY, NOREPLY, "first")])
    subprocess.run(["git", "tag", "-a", "t1", "-m", "release"], cwd=root, check=True,
                   env=_env(committer=PERSONAL))
    assert [(o, f) for o, f, _ in personal_emails_in_history(root)] == [("t1", "tagger")]


def test_a_commit_reachable_only_from_a_tag_is_checked(tmp_path):
    root = _repo(tmp_path, [(NOREPLY, NOREPLY, "first")])
    subprocess.run(["git", "checkout", "-q", "-b", "side"], cwd=root, check=True, env=_env())
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "side"], cwd=root,
                   check=True, env=_env(author=PERSONAL))
    subprocess.run(["git", "tag", "t1"], cwd=root, check=True, env=_env())
    subprocess.run(["git", "checkout", "-q", "main"], cwd=root, check=True, env=_env())
    assert [f for _, f, _ in personal_emails_in_history(root)] == ["author"]


def test_a_shallow_clone_is_refused(tmp_path):
    src = _repo(tmp_path / "src", [(NOREPLY, NOREPLY, "first"), (NOREPLY, NOREPLY, "second")])
    dst = tmp_path / "dst"
    subprocess.run(["git", "clone", "-q", "--depth", "1", src.as_uri(), str(dst)],
                   check=True, env=_env())
    with pytest.raises(RuntimeError, match="shallow"):
        history(dst)


def test_a_failure_message_does_not_print_the_address():
    assert PERSONAL.split("@")[0] not in mask(PERSONAL)
    assert mask(PERSONAL).endswith("@gmail.com")


def test_an_address_in_a_file_with_a_non_ascii_name_is_found(tmp_path):
    root = _repo(tmp_path, [])
    (root / "résumé.md").write_text(f"contact: {PERSONAL}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True, env=_env())
    found, unreadable = personal_emails_in_files(root)
    assert unreadable == {} and list(found) == ["résumé.md"]


def test_a_tracked_pdf_is_reported_not_skipped(tmp_path):
    root = _repo(tmp_path, [])
    (root / "paper.pdf").write_bytes(b"%PDF-1.4\n\x00\x01 compressed")
    subprocess.run(["git", "add", "."], cwd=root, check=True, env=_env())
    assert list(personal_emails_in_files(root)[1]) == ["paper.pdf"]


@pytest.mark.parametrize("local,domain", [
    ("first.last.ml", "gmail.com"),
    ("someone", "university.ac.jp"),
    ("First.Last+tag", "sub.domain.co.uk"),
])
def test_a_reachable_address_is_caught(local, domain):
    addr = _addr(local, domain)
    assert personal_emails(f"contact: {addr}") == [addr]


@pytest.mark.parametrize("local,domain", [
    ("52999158+SomeUser", "users.noreply.github.com"),
    ("noreply", "github.com"),
    ("noreply", "anthropic.com"),
    ("anonymous", "example.org"),
    ("anon", "cs.example.org"),
    ("Anon", "Example.ORG"),
])
def test_an_unreachable_address_is_allowed(local, domain):
    assert personal_emails(f"Author <{_addr(local, domain)}>") == []


@pytest.mark.parametrize("local,domain", [
    # The allowed name only in the middle of a real domain.
    ("person", "users.noreply.github.com.evil.jp"),
    ("person", "example.org.evil.jp"),
    ("person", "notexample.com"),
    # One address is allowed, not its domain.
    ("someone", "anthropic.com"),
    ("someone", "github.com"),
])
def test_an_allowed_name_inside_a_reachable_address_is_still_caught(local, domain):
    addr = _addr(local, domain)
    assert personal_emails(f"contact: {addr}") == [addr]


def test_a_metric_name_is_not_an_address():
    # The docs write `F1@0.5` a lot. A gate that fires on it stops being
    # believed.
    assert personal_emails("inst-F1@0.5 and F1@0.75") == []


def test_a_file_that_is_not_utf8_is_reported(tmp_path):
    p = tmp_path / "utf16.md"
    p.write_bytes(PERSONAL.encode("utf-16"))
    with pytest.raises(ValueError):
        read_text_strictly(p)


def test_a_file_with_a_broken_byte_is_reported(tmp_path):
    p = tmp_path / "broken.md"
    p.write_bytes(b"contact \xff\xfe")
    with pytest.raises(ValueError):
        read_text_strictly(p)


def test_a_utf8_file_reads_back(tmp_path):
    # Without this, the two above would also pass on a reader that always
    # raises.
    p = tmp_path / "ok.md"
    p.write_text("naïve and ASCII\n", encoding="utf-8")
    assert read_text_strictly(p) == "naïve and ASCII\n"
