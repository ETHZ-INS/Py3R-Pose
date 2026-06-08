"""
Filter chain syntax parser.

Current grammar (single-stream):

    filter_chain   := filter_step (PIPE filter_step)*
    filter_step    := NAME subject_sel? context_sel? LPAREN param_list? RPAREN
    subject_sel    := LANGLE selector_list RANGLE
    context_sel    := LBRACKET selector_list RBRACKET
    selector_list  := selector (COMMA selector)*
    selector       := 'i:' type_name            # instance type (explicit prefix)
                    | type_name                  # instance type (shorthand)
    param_list     := param (COMMA param)*
    param          := NAME EQUALS value          # named param
                    | value                      # positional param
    value          := LBRACKET NAME (COMMA NAME)* RBRACKET  # type name list
                    | NUMBER
                    | NAME
                    | BOOL

Examples:
    confidence<mouse>(0.5, 0.3) | arena<mouse>[epm_arena](0.1)
    instance_type([mouse, epm_arena]) | median<mouse>()
    confidence<i:mouse>(instance_threshold=0.7)

Planned additions (multi-stream graph model, not yet implemented):
    - Stream selectors inside <> and []:  's:stream_name'
      e.g. arena<s:tracking, i:mouse>[s:arena_cam](0.1)
    - Output stream naming:  '@name' suffix on a step
      e.g. confidence(0.5) @filtered
    Semantics: <s:...> declares input source stream(s); @name declares the
    output stream name (default: replaces input). Steps with no stream
    qualifier read from the implicit current stream (output of the previous
    step, or 'main' for the first step).
"""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class _T:
    NAME     = "NAME"
    NUMBER   = "NUMBER"
    BOOL     = "BOOL"
    LANGLE   = "LANGLE"    # <
    RANGLE   = "RANGLE"    # >
    LBRACKET = "LBRACKET"  # [
    RBRACKET = "RBRACKET"  # ]
    LPAREN   = "LPAREN"    # (
    RPAREN   = "RPAREN"    # )
    COLON    = "COLON"     # :
    COMMA    = "COMMA"     # ,
    PIPE     = "PIPE"      # |
    EQUALS   = "EQUALS"    # =
    EOF      = "EOF"


@dataclass
class Token:
    type: str
    value: Any
    pos: int


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class FilterSyntaxError(ValueError):
    def __init__(self, message: str, pos: int, source: str = ""):
        if source:
            arrow = " " * pos + "^"
            super().__init__(f"{message}\n  {source}\n  {arrow}")
        else:
            super().__init__(f"{message} (position {pos})")
        self.pos = pos


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_PATTERNS = [
    (re.compile(r"\s+"),                           None),
    (re.compile(r"\|"),                            _T.PIPE),
    (re.compile(r"<"),                             _T.LANGLE),
    (re.compile(r">"),                             _T.RANGLE),
    (re.compile(r"\["),                            _T.LBRACKET),
    (re.compile(r"]"),                             _T.RBRACKET),
    (re.compile(r"\("),                            _T.LPAREN),
    (re.compile(r"\)"),                            _T.RPAREN),
    (re.compile(r":"),                             _T.COLON),
    (re.compile(r","),                             _T.COMMA),
    (re.compile(r"="),                             _T.EQUALS),
    (re.compile(r"[+-]?\d+\.\d+"),                _T.NUMBER),   # float before int
    (re.compile(r"[+-]?\d+"),                      _T.NUMBER),
    (re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*"),       _T.NAME),
]


def _tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        for pattern, tok_type in _TOKEN_PATTERNS:
            m = pattern.match(text, pos)
            if m:
                if tok_type is not None:
                    raw = m.group(0)
                    if tok_type == _T.NAME and raw.lower() in ("true", "false"):
                        tokens.append(Token(_T.BOOL, raw.lower() == "true", pos))
                    elif tok_type == _T.NUMBER:
                        val: Union[int, float] = float(raw) if "." in raw else int(raw)
                        tokens.append(Token(_T.NUMBER, val, pos))
                    else:
                        tokens.append(Token(tok_type, raw, pos))
                pos = m.end()
                break
        else:
            raise FilterSyntaxError(f"Unexpected character '{text[pos]}'", pos, text)
    tokens.append(Token(_T.EOF, None, pos))
    return tokens


# ---------------------------------------------------------------------------
# AST dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SelectorSpec:
    """Parsed content of a <...> or [...] selector block."""
    instance_type_names: List[str] = field(default_factory=list)


# A raw param is (name_or_None, value) — name is None for positional params.
RawParam = Tuple[Optional[str], Any]


@dataclass
class FilterStepSpec:
    name: str
    subject_selector: Optional[SelectorSpec]   # from <...>, or None
    context_selector: Optional[SelectorSpec]   # from [...], or None
    params: List[RawParam]
    pos: int                                   # source position for error messages


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: List[Token], source: str):
        self._tokens = tokens
        self._pos = 0
        self._source = source

    # -- token helpers -------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _consume(self, expected_type: Optional[str] = None) -> Token:
        tok = self._tokens[self._pos]
        if expected_type and tok.type != expected_type:
            raise FilterSyntaxError(
                f"Expected {expected_type} but got '{tok.value}'",
                tok.pos, self._source,
            )
        self._pos += 1
        return tok

    def _match(self, *types: str) -> bool:
        return self._peek().type in types

    def _peek_ahead(self, offset: int) -> Token:
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return self._tokens[-1]  # EOF

    # -- grammar rules -------------------------------------------------------

    def parse(self) -> List[FilterStepSpec]:
        steps = [self._parse_step()]
        while self._match(_T.PIPE):
            self._consume(_T.PIPE)
            steps.append(self._parse_step())
        self._consume(_T.EOF)
        return steps

    def _parse_step(self) -> FilterStepSpec:
        pos = self._peek().pos
        name_tok = self._consume(_T.NAME)

        subject_sel: Optional[SelectorSpec] = None
        if self._match(_T.LANGLE):
            subject_sel = self._parse_selector(_T.LANGLE, _T.RANGLE)

        context_sel: Optional[SelectorSpec] = None
        if self._match(_T.LBRACKET):
            context_sel = self._parse_selector(_T.LBRACKET, _T.RBRACKET)

        self._consume(_T.LPAREN)
        params: List[RawParam] = []
        if not self._match(_T.RPAREN):
            params = self._parse_param_list()
        self._consume(_T.RPAREN)

        return FilterStepSpec(name_tok.value, subject_sel, context_sel, params, pos)

    def _parse_selector(self, open_type: str, close_type: str) -> SelectorSpec:
        self._consume(open_type)
        spec = SelectorSpec()
        if not self._match(close_type):
            self._parse_selector_item(spec)
            while self._match(_T.COMMA):
                self._consume(_T.COMMA)
                self._parse_selector_item(spec)
        self._consume(close_type)
        return spec

    def _parse_selector_item(self, spec: SelectorSpec) -> None:
        name_tok = self._consume(_T.NAME)
        if self._match(_T.COLON):
            self._consume(_T.COLON)
            value_tok = self._consume(_T.NAME)
            if name_tok.value == "s":
                raise FilterSyntaxError(
                    f"Stream qualifiers (s:{value_tok.value}) are not yet supported. "
                    "Planned for future multi-stream support.",
                    name_tok.pos, self._source,
                )
            elif name_tok.value == "i":
                spec.instance_type_names.append(value_tok.value)
            else:
                raise FilterSyntaxError(
                    f"Unknown selector prefix '{name_tok.value}': use 'i:' for instance types",
                    name_tok.pos, self._source,
                )
        else:
            # bare name = instance type shorthand
            spec.instance_type_names.append(name_tok.value)

    def _parse_param_list(self) -> List[RawParam]:
        params = [self._parse_param()]
        while self._match(_T.COMMA):
            self._consume(_T.COMMA)
            params.append(self._parse_param())
        return params

    def _parse_param(self) -> RawParam:
        # named param: NAME EQUALS value
        if self._match(_T.NAME) and self._peek_ahead(1).type == _T.EQUALS:
            name_tok = self._consume(_T.NAME)
            self._consume(_T.EQUALS)
            return (name_tok.value, self._parse_value())
        return (None, self._parse_value())

    def _parse_value(self) -> Any:
        if self._match(_T.LBRACKET):
            # list of type names: [mouse, epm_arena]
            self._consume(_T.LBRACKET)
            items: List[str] = []
            if not self._match(_T.RBRACKET):
                items.append(self._consume(_T.NAME).value)
                while self._match(_T.COMMA):
                    self._consume(_T.COMMA)
                    items.append(self._consume(_T.NAME).value)
            self._consume(_T.RBRACKET)
            return items
        if self._match(_T.NUMBER):
            return self._consume(_T.NUMBER).value
        if self._match(_T.BOOL):
            return self._consume(_T.BOOL).value
        if self._match(_T.NAME):
            return self._consume(_T.NAME).value
        tok = self._peek()
        raise FilterSyntaxError(
            f"Expected a value (number, name, true/false, or [list]) but got '{tok.value}'",
            tok.pos, self._source,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_filter_chain(text: str) -> List[FilterStepSpec]:
    """
    Parse a filter chain string into a list of FilterStepSpec objects.
    Raises FilterSyntaxError with position information on invalid input.
    """
    tokens = _tokenize(text)
    return _Parser(tokens, text).parse()

