from photoreel.concept import interpret


def test_style_from_mood_words():
    assert interpret("잔잔하고 따뜻한 여행 기록").style == "calm"
    assert interpret("신나는 하이라이트!").style == "upbeat"
    assert interpret("영화 같은 웅장한 분위기").style == "cinematic"
    assert interpret("필름 느낌의 추억").style == "nostalgic"
    assert interpret("A calm, cozy recap").style == "calm"


def test_no_clue_falls_back_and_says_so():
    r = interpret("우리 집 고양이")
    assert r.style == "clean" and r.style_matched is False
    assert "animal" in r.emphasis  # 분위기 단서는 없지만 강조 대상은 읽음


def test_modifiers():
    r = interpret("잔잔하게, 시간순으로, 흑백, 음악 없이, 40초 정도")
    assert (r.style, r.order, r.grade, r.music, r.duration) == ("calm", "chronological", "bw", False, 40.0)
    assert interpret("1분 30초짜리").duration == 90.0
    assert interpret("천천히 보여줘").pace > 1
    assert interpret("담백하게 빠르게").pace < 1


def test_generative_requests_are_reported_not_faked():
    r = interpret("지브리 애니메이션 스타일로, 파도가 치게 움직이게 해줘")
    assert len(r.unsupported) == 2
    assert all("적용되지 않았습니다" in u for u in r.unsupported)


def test_empty():
    r = interpret("")
    assert r.style == "clean" and not r.signals and not r.unsupported
