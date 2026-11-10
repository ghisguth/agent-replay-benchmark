"""Display width and formatting tests: ASCII, CJK, emoji with VS16, Nerd Font PUA glyphs, ANSI escapes, number formatting.

    python3 -m unittest tests.test_fmt
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import chat, fmt, report


class TestCharWidth(unittest.TestCase):
    def test_ascii(self):
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 `~!@#$%^&*()-_=+[]{}|;:'\",.<>/?":
            self.assertEqual(fmt.cw(ch), 1, f"ASCII {ch!r} should have width 1")

    def test_cjk(self):
        cjk_chars = ["中", "文", "日", "本", "語", "한", "글", "，", "。"]
        for ch in cjk_chars:
            self.assertEqual(fmt.cw(ch), 2, f"CJK {ch!r} should have width 2")

    def test_nerd_font_glyphs(self):
        pua_chars = ["\uf085", "\uf120", "\uf00c", "\ue000", "\uf8ff", "\U000f0000", "\U000f0001"]
        for ch in pua_chars:
            self.assertEqual(fmt.cw(ch), 1, f"PUA {ch!r} should have width 1 in standard cw")
            self.assertEqual(fmt.live_cw(ch), 2, f"PUA {ch!r} should have width 2 in live_cw")

    def test_combining_and_zero_width(self):
        self.assertEqual(fmt.cw("\u0300"), 0)  # combining grave accent
        self.assertEqual(fmt.cw("\u0301"), 0)  # combining acute accent
        self.assertEqual(fmt.cw("\ufe0f"), 0)  # VS16
        self.assertEqual(fmt.cw("\ufe0e"), 0)  # VS15
        self.assertEqual(fmt.cw("\u200b"), 0)  # zero-width space
        self.assertEqual(fmt.cw("\u200d"), 0)  # zero-width joiner
        self.assertEqual(fmt.cw("\ufeff"), 0)  # zero-width no-break space
        self.assertEqual(fmt.cw(""), 0)


class TestStringWidth(unittest.TestCase):
    def test_ascii_vlen(self):
        self.assertEqual(fmt.vlen(""), 0)
        self.assertEqual(fmt.vlen("hello"), 5)
        self.assertEqual(fmt.vlen("hello world"), 11)

    def test_cjk_vlen(self):
        self.assertEqual(fmt.vlen("你好"), 4)
        self.assertEqual(fmt.vlen("Hello 世界!"), 11)  # 6 ascii + 4 CJK + 1 punct

    def test_emoji_and_vs16(self):
        # Naturally wide emojis
        self.assertEqual(fmt.vlen("🚀"), 2)
        self.assertEqual(fmt.vlen("🎉🔥"), 4)
        self.assertEqual(fmt.vlen("🚀\ufe0f"), 2)  # VS16 after wide emoji does not add extra width

        # Text-presentation symbols: width 1 without VS16, width 2 with VS16
        self.assertEqual(fmt.vlen("⚙"), 1)
        self.assertEqual(fmt.vlen("⚙\ufe0f"), 2)
        self.assertEqual(fmt.vlen("☀"), 1)
        self.assertEqual(fmt.vlen("☀\ufe0f"), 2)
        self.assertEqual(fmt.vlen("✔"), 1)
        self.assertEqual(fmt.vlen("✔\ufe0f"), 2)
        self.assertEqual(fmt.vlen("⚠"), 1)
        self.assertEqual(fmt.vlen("⚠\ufe0f"), 2)

        # Isolated VS16
        self.assertEqual(fmt.vlen("\ufe0f"), 0)

        # Mixed sentence with multiple VS16 emojis
        self.assertEqual(fmt.vlen("gear ⚙\ufe0f and sun ☀\ufe0f"), 5 + 2 + 9 + 2)

    def test_nerd_font_vlen(self):
        self.assertEqual(fmt.vlen("\uf085"), 1)
        self.assertEqual(fmt.vlen("bash \uf120 run"), 10)
        # live_vlen counts PUA as width 2
        self.assertEqual(fmt.live_vlen("\uf085"), 2)
        self.assertEqual(fmt.live_vlen("bash \uf120 run"), 11)

    def test_ansi_escapes(self):
        self.assertEqual(fmt.vlen("\x1b[31mhello\x1b[0m"), 5)
        self.assertEqual(fmt.vlen("\x1b[1;32;40mGreen text\x1b[0m"), 10)
        self.assertEqual(fmt.vlen("\x1b[33m你好\x1b[0m"), 4)
        self.assertEqual(fmt.vlen("\x1b[2K"), 0)


class TestFormattingAndFitting(unittest.TestCase):
    def test_clip(self):
        self.assertEqual(fmt.clip("hello world", 15), "hello world")
        self.assertEqual(fmt.clip("hello world", 5), "hell…")
        self.assertEqual(fmt.clip("你好世界", 6), "你好…")  # 4 cols for 你好 + 1 for … <= 6
        self.assertEqual(fmt.clip("   many   spaces   here   ", 20), "many spaces here")

    def test_fit(self):
        self.assertEqual(fmt.fit("hello", 8, align="l"), "hello   ")
        self.assertEqual(fmt.fit("hello", 8, align="r"), "   hello")
        self.assertEqual(fmt.fit("hello world", 5), "hell…")
        # mid ellipsis
        mid = fmt.fit("F428FalsifyP1213R2", 12, mid=True)
        self.assertEqual(fmt.vlen(mid), 12)
        self.assertIn("…", mid)
        self.assertTrue(mid.startswith("F428F") and mid.endswith("1213R2"[-4:]))

    def test_hard_wrap(self):
        self.assertEqual(fmt.hard_wrap("1234567890", 4), ["1234", "5678", "90"])
        self.assertEqual(fmt.hard_wrap("你好世界", 4), ["你好", "世界"])

    def test_fmt_number(self):
        self.assertEqual(fmt.fmt_number(7), "7")
        self.assertEqual(fmt.fmt_number(999), "999")
        self.assertEqual(fmt.fmt_number(1000), "1K")
        self.assertEqual(fmt.fmt_number(1234), "1.2K")
        self.assertEqual(fmt.fmt_number(9999), "10K")
        self.assertEqual(fmt.fmt_number(12345), "12K")
        self.assertEqual(fmt.fmt_number(123456), "123K")
        self.assertEqual(fmt.fmt_number(1500000), "1.5M")

    def test_fmt_clock(self):
        self.assertEqual(fmt.fmt_clock(5), " 5s")
        self.assertEqual(fmt.fmt_clock(12), "12s")
        self.assertEqual(fmt.fmt_clock(68), "1m08s")
        self.assertEqual(fmt.fmt_clock(600), "10m00s")
        self.assertEqual(fmt.fmt_clock(3720), "1h02m")

    def test_fmt_duration(self):
        self.assertEqual(fmt.fmt_duration(0), "0ms")
        self.assertEqual(fmt.fmt_duration(450), "450ms")
        self.assertEqual(fmt.fmt_duration(8200), "8.2s")
        self.assertEqual(fmt.fmt_duration(65000), "1m05s")
        self.assertEqual(fmt.fmt_duration(3 * 3600000 + 120000), "3h02m")
        self.assertEqual(fmt.fmt_duration(5 * 86400000), "5d00h")


class TestReExports(unittest.TestCase):
    def test_chat_reexports(self):
        self.assertIs(chat.cw, fmt.cw)
        self.assertIs(chat.vlen, fmt.vlen)
        self.assertIs(chat.live_vlen, fmt.live_vlen)
        self.assertIs(chat.clip, fmt.clip)
        self.assertIs(chat.hard_wrap, fmt.hard_wrap)
        self.assertIs(chat.fmt_number, fmt.fmt_number)
        self.assertIs(chat.fmt_clock, fmt.fmt_clock)
        self.assertIs(chat.fmt_duration, fmt.fmt_duration)
        self.assertIs(chat.term_width, fmt.term_width)

    def test_report_reexports(self):
        self.assertIs(report._cw, fmt.cw)
        self.assertIs(report.vlen, fmt.vlen)
        self.assertIs(report.fit, fmt.fit)
        self.assertIs(report.term_limit, fmt.term_limit)


class Formatting(unittest.TestCase):
    def test_durations_one_style(self):
        self.assertEqual([fmt.ms(x) for x in (896, 8574, 24550, 7306, 70600)], ["0.90s", "8.57s", "24.6s", "7.31s", "1m11s"])
        self.assertEqual(fmt.secs(None), "–")

    def test_thousands_separators(self):
        self.assertEqual((fmt.num(12174), fmt.num(None), fmt.num(837)), ("12,174", "–", "837"))

    def test_short_names_strip_the_common_prefix(self):
        self.assertEqual(fmt.short_names(["ISTA-DASLab/Qwen3.8-GSQ-RCO-GGUF", "ISTA-DASLab/Qwen3.8-GSQ-Q4"]),
                         ["RCO-GGUF", "Q4"])
        self.assertEqual(fmt.short_names(["alpha", "beta"]), ["alpha", "beta"])
        self.assertEqual(fmt.short_names(["only"]), ["only"])
        self.assertEqual(fmt.short_names(["m-", "m-x"]), ["m-", "m-x"])  # nothing may become empty


if __name__ == "__main__":
    unittest.main()
