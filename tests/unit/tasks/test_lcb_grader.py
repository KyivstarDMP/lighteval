# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The LiveCodeBench stdin/stdout grader accepts every standard way a Python program reads and writes."""

import pytest

from lighteval.tasks.tasks.lcb.codegen_metrics import grade_stdio


INPUTS = ["2\n3 4\n", "3\n1 2 3\n"]
OUTPUTS = ["7\n", "6\n"]

PROGRAMS = {
    "input": "n = int(input())\nprint(sum(map(int, input().split())))",
    "stdin.read": "import sys\ndata = sys.stdin.read().split()\nprint(sum(map(int, data[1:])))",
    "stdin.readline": "import sys\nsys.stdin.readline()\nprint(sum(map(int, sys.stdin.readline().split())))",
    # 2026-09-30: the mocked stdin had no buffer, so every program reading it crashed (Qwen3.8-27B LCB 0.53 -> 0.90)
    "stdin.buffer.read": "import sys\ndata = sys.stdin.buffer.read().split()\nprint(sum(map(int, data[1:])))",
    "stdin.buffer.readline": (
        "import sys\nsys.stdin.buffer.readline()\nprint(sum(map(int, sys.stdin.buffer.readline().split())))"
    ),
    "stdout.buffer.write": (
        "import sys\ndata = sys.stdin.buffer.read().split()\n"
        "sys.stdout.buffer.write(str(sum(map(int, data[1:]))).encode() + b'\\n')"
    ),
    "__main__ guard": (
        "import sys\n\ndef main():\n    data = sys.stdin.buffer.read().split()\n"
        "    sys.stdout.write(str(sum(map(int, data[1:]))) + '\\n')\n\n"
        "if __name__ == '__main__':\n    main()"
    ),
}


@pytest.mark.parametrize("program", PROGRAMS.values(), ids=PROGRAMS.keys())
def test_every_io_idiom_passes_a_correct_program(program):
    assert grade_stdio(program, INPUTS, OUTPUTS, timeout=6) == [True, True]


def test_text_and_bytes_written_to_stdout_are_captured_in_order():
    program = "import sys\nprint('a')\nsys.stdout.buffer.write(b'b\\n')\nprint('c')"
    assert grade_stdio(program, [""], ["a\nb\nc\n"], timeout=6) == [True]


def test_a_wrong_answer_still_fails():
    program = "import sys\nsys.stdout.buffer.write(b'0\\n')"
    assert grade_stdio(program, INPUTS, OUTPUTS, timeout=6) != [True, True]
