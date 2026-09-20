from pronunciation_oracle.subtitles import parse_srt, parse_subtitles, parse_vtt

SRT_SAMPLE = """1
00:00:01,000 --> 00:00:03,500
Hey, Pikachu!

2
00:00:04,000 --> 00:00:06,250
Use Thunderbolt now!

3
00:00:07,000 --> 00:00:09,000
ASH: I choose you, Pikachu!
"""

VTT_SAMPLE = """WEBVTT

1
00:00:01.000 --> 00:00:03.500
Hey, <i>Pikachu</i>!

2
00:00:04.000 --> 00:00:06.250 align:start
Use Thunderbolt now!
"""


def test_parse_srt_basic():
    lines = parse_srt(SRT_SAMPLE)
    assert len(lines) == 3
    assert lines[0].start == 1.0
    assert lines[0].end == 3.5
    assert lines[0].text == "Hey, Pikachu!"
    assert lines[1].start == 4.0
    assert lines[1].end == 6.25


def test_parse_srt_strips_speaker_label():
    lines = parse_srt(SRT_SAMPLE)
    assert lines[2].text == "I choose you, Pikachu!"


def test_parse_vtt_basic_and_strips_tags():
    lines = parse_vtt(VTT_SAMPLE)
    assert len(lines) == 2
    assert lines[0].text == "Hey, Pikachu!"
    assert lines[0].start == 1.0
    assert lines[0].end == 3.5


def test_parse_vtt_ignores_cue_settings_after_timestamp():
    lines = parse_vtt(VTT_SAMPLE)
    assert lines[1].start == 4.0
    assert lines[1].end == 6.25
    assert lines[1].text == "Use Thunderbolt now!"


def test_parse_subtitles_dispatches_on_extension(tmp_path):
    srt_path = tmp_path / "sub.srt"
    srt_path.write_text(SRT_SAMPLE, encoding="utf-8")
    vtt_path = tmp_path / "sub.vtt"
    vtt_path.write_text(VTT_SAMPLE, encoding="utf-8")

    assert len(parse_subtitles(srt_path)) == 3
    assert len(parse_subtitles(vtt_path)) == 2


def test_parse_srt_handles_no_blank_line_at_end():
    text = "1\n00:00:00,000 --> 00:00:01,000\nHi\n"
    lines = parse_srt(text)
    assert len(lines) == 1
    assert lines[0].text == "Hi"
