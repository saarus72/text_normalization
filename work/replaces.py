from typing import Optional
from difflib import SequenceMatcher
import re


class Replace(dict):
    """
    `type` is "E" for equal if `text_to` is None or is the same with `text_from`.
    Otherwise `type` is "R" for replace.
    """

    def __init__(
        self,
        text_from: str, text_to: Optional[str]=None,
        *args, **kwargs
    ):
        """
        If `text_to` is None the Replace is supposed to be equal.
        """
        super().__init__(*args, **kwargs)
        self["text_from"] = text_from
        self["text_to"] = text_to

    @property
    def type(self):
        return "E" if (self.text_from == self.text_to or self["text_to"] is None) else "R"
        
    @property
    def text_from(self):
        return self["text_from"]
    @text_from.setter
    def text_from(self, value):
        self["text_from"] = value

    @property
    def text_to(self):
        return self["text_to"] if self["text_to"] is not None else self.text_from
    @text_to.setter
    def text_to(self, value):
        self["text_to"] = value

    def extend(self, r):
        if self.type != r.type:
            raise Exception("Replace type mismatch")
        self.text_from += r.text_from
        self.text_to += r.text_to
        return self


class Replaces(list):
    __re_digits_latin = re.compile(r"[a-zA-Z\d]")
    
    def __init__(self, *args):
        super().__init__(*args)
        for i, elem in enumerate(self):
            if not isinstance(elem, Replace):
                if isinstance(elem, dict):
                    self[i] = Replace(**elem)
                else:
                    self[i] = Replace(elem)
    
    def add(self, r: Replace):
        if self and r.type == self[-1].type:
            self[-1].extend(r)
        else:
            return super().append(r)

    def __repr__(self):
        return "\n".join((f'{r.type}|{r.text_from}{" => " + r.text_to if r.type != "E" else ""}' for r in self))

    @staticmethod
    def from_sequences(seq1, seq2, ingore_not_digit_latin=True):
        """
        If `ingore_not_digit_latin` element pairs containing no digits and latin would be treated as equal to `seq1` element.
        """
        sm = SequenceMatcher(
            # lambda x: not re.search(r"\w", x.strip()),
            a=seq1,
            b=seq2,
            autojunk=False
        )
        result = Replaces()
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            text_from, text_to = "".join(seq1[i1:i2]), "".join(seq2[j1:j2])
            if tag == "equal":
                pass
            elif tag == "replace" and "".join((_.strip() for _ in seq1[i1:i2])) == "".join((_.strip() for _ in seq2[j1:j2])):
                text_to = None
            elif ingore_not_digit_latin and \
                    not re.search(Replaces.__re_digits_latin, text_from) and \
                    not re.search(Replaces.__re_digits_latin, text_to):
                text_to = None
            result.add(Replace(text_from, text_to))
        return result
