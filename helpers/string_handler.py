import re
import ast
import json

def fix_unescaped_inner_quotes(raw: str) -> str:
    """
    Convert a string that looks like a Python dict/list written with single quotes
    to real JSON (double-quoted) while keeping inner apostrophes intact.
    Returns the JSON *string*; call json.loads(...) if you need the object.
    """
    OUT, IN_SINGLE = 0, 1          # two parser states
    state = OUT
    out = []

    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]

        if state == OUT:           # -------- not inside a single-quoted string
            if ch == "'":
                out.append('"')    # opening quote turns into "
                state = IN_SINGLE
            else:
                out.append(ch)

        else:                      # -------- inside a single-quoted string
            if ch == "\\":         # keep existing escapes unchanged
                out.append(ch)
                if i + 1 < n:
                    out.append(raw[i + 1])
                    i += 1
            elif ch == "'":
                # If next char is a letter/number it's an apostrophe (keep it);
                # otherwise it’s the closing quote.
                if (i + 1 < n and raw[i + 1].isalnum()):
                    out.append(ch)          # apostrophe inside the value
                else:
                    out.append('"')         # closing quote becomes "
                    state = OUT
            elif ch == '"':                 # double quote **inside** value
                out.append(r'\"')           # must be escaped for JSON
            else:
                out.append(ch)

        i += 1

    return ''.join(out)

def escape_inner_double_quotes(s):
    result = []
    in_string = False
    escape_next = False

    for i, c in enumerate(s):
        if escape_next:
            result.append(c)
            escape_next = False
            continue

        if c == '\\':
            result.append(c)
            escape_next = True
            continue

        if c == '"':
            if in_string:
                # Look ahead to decide if this is an internal quote
                j = i + 1
                while j < len(s) and s[j].isspace():
                    j += 1
                if j < len(s) and s[j] != ':' and s[j] != ',' and s[j] != '}':
                    # It's likely an internal quote, escape it
                    result.append(r'\"')
                    continue
                else:
                    # It's closing the string
                    in_string = False
                    result.append('"')
            else:
                in_string = True
                result.append('"')
        else:
            result.append(c)

    return ''.join(result)