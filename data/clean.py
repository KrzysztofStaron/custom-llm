import re

_ALLOWED = re.compile(r"^[\t\n\r -~]*$")
_FENCED_CODE = re.compile(r"```[^\n]*\n.*?```|```[^\n]*```", re.DOTALL)
_TABLE_SEPARATOR = re.compile(r"^\|[\s\-:|]+\|(?:[\s\-:|]*\|)*$")


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.count("|") < 2:
        return False
    if _TABLE_SEPARATOR.fullmatch(stripped):
        return True
    if re.fullmatch(r"[\-\|: ]+", stripped) and "-" in stripped:
        return True
    return stripped.startswith("|")


def strip_markdown(text: str) -> str:
    text = _FENCED_CODE.sub("", text)
    text = re.sub(r"^```[^\n]*$", "", text, flags=re.MULTILINE)
    lines = [line for line in text.split("\n") if not _is_table_line(line)]
    return "\n".join(lines)


def clean_text(text: str) -> str:
    text = strip_markdown(text)
    text = re.sub(r"[^\t\n\r -~]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_unwanted_chars(text: str) -> bool:
    return _ALLOWED.fullmatch(text) is None
