import time

from fastapi.testclient import TestClient

from photoreel.server import app

client = TestClient(app)


def upload(pid, jpeg, n=4):
    from datetime import datetime, timedelta

    t0 = datetime(2026, 3, 1, 10)
    for i in range(n):
        data = jpeg(i + 1, (640, 480) if i % 2 else (480, 640), t0 + timedelta(minutes=20 * (n - i)))
        r = client.post(f"/api/projects/{pid}/photos", files={"files": (f"img{i}.jpg", data, "image/jpeg")})
        assert r.status_code == 200 and len(r.json()["added"]) == 1


def test_full_flow_upload_plan_edit_render_download(jpeg):
    pid = client.post("/api/projects").json()["id"]
    upload(pid, jpeg)

    plan = client.post(f"/api/projects/{pid}/plan", json={"concept": "잔잔한 기록", "title": "제목"}).json()
    names = {p["id"]: p["filename"] for p in client.get(f"/api/projects/{pid}").json()["photos"]}
    assert [names[s["photo_id"]] for s in plan["scenes"]] == ["img3.jpg", "img2.jpg", "img1.jpg", "img0.jpg"]
    assert plan["style"] == "calm" and plan["engine"] == "edit"

    # 화면에서 하는 수정: 한 장 빼고, 순서를 바꾸고, 작은 크기로 (테스트를 빠르게)
    plan["scenes"][1]["included"] = False
    plan["scenes"][0], plan["scenes"][2] = plan["scenes"][2], plan["scenes"][0]
    for s in plan["scenes"]:
        s["beats"] = 2
    plan["output"] = {"aspect": "custom", "width": 240, "height": 426, "fps": 12}
    saved = client.put(f"/api/projects/{pid}/plan", json=plan)
    assert saved.status_code == 200
    saved = saved.json()
    assert [names[s["photo_id"]] for s in saved["scenes"] if s["included"]] == ["img1.jpg", "img3.jpg", "img0.jpg"]
    assert saved["scenes"][-1]["included"] is False

    assert client.post(f"/api/projects/{pid}/render").status_code == 200
    for _ in range(300):
        job = client.get(f"/api/projects/{pid}/render").json()
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done", job
    n = job["render"]
    video = client.get(f"/api/projects/{pid}/renders/{n}/video?download=1")
    assert video.status_code == 200 and video.content[4:8] == b"ftyp"
    assert "attachment" in video.headers["content-disposition"]
    part = client.get(f"/api/projects/{pid}/renders/{n}/video", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100  # 브라우저 재생·탐색에 필요

    renders = client.get(f"/api/projects/{pid}").json()["renders"]
    assert renders[-1]["status"] == "ok" and renders[-1]["scenes"] == 3 and renders[-1]["cost_usd"] == 0.0


def test_rejects_bad_input(jpeg):
    pid = client.post("/api/projects").json()["id"]
    assert client.post(f"/api/projects/{pid}/plan", json={"concept": "x"}).status_code == 400  # 사진 없음
    r = client.post(f"/api/projects/{pid}/photos", files={"files": ("note.txt", b"hello", "text/plain")})
    assert r.json()["added"] == [] and r.json()["errors"]
    r = client.post(f"/api/projects/{pid}/photos", files={"files": ("broken.jpg", b"not a jpeg", "image/jpeg")})
    assert r.json()["added"] == [] and "읽지 못함" in r.json()["errors"][0]["error"]
    assert client.get("/api/projects/zzzz").status_code == 404
    assert client.get("/api/projects/../../etc").status_code == 404

    upload(pid, jpeg, 2)
    plan = client.post(f"/api/projects/{pid}/plan", json={"concept": ""}).json()
    bad = dict(plan, scenes=plan["scenes"] + [dict(plan["scenes"][0])])
    assert client.put(f"/api/projects/{pid}/plan", json=bad).status_code == 400  # 같은 사진 두 번
    bad = dict(plan, engine="generate")
    assert client.put(f"/api/projects/{pid}/plan", json=bad).status_code == 200
    assert client.post(f"/api/projects/{pid}/render").status_code == 400  # 연결 안 된 엔진은 실행 거부
    bad = dict(plan, scenes=[dict(plan["scenes"][0], motion="teleport")])
    assert client.put(f"/api/projects/{pid}/plan", json=bad).status_code == 400
    assert client.post(f"/api/projects/{pid}/plan", json={"concept": "", "aspect": "3:7"}).status_code == 400


def test_delete_photo_updates_plan(jpeg):
    pid = client.post("/api/projects").json()["id"]
    upload(pid, jpeg, 3)
    plan = client.post(f"/api/projects/{pid}/plan", json={"concept": ""}).json()
    victim = plan["scenes"][0]["photo_id"]
    assert client.delete(f"/api/projects/{pid}/photos/{victim}").status_code == 200
    after = client.get(f"/api/projects/{pid}").json()
    assert victim not in [p["id"] for p in after["photos"]]
    assert victim not in [s["photo_id"] for s in after["plan"]["scenes"]]
    assert client.get(f"/api/projects/{pid}/photos/{victim}/thumb").status_code == 404


def test_style_change_keeps_order_exclusions_and_engine(jpeg):
    pid = client.post("/api/projects").json()["id"]
    upload(pid, jpeg, 4)
    plan = client.post(f"/api/projects/{pid}/plan", json={"concept": "잔잔하게"}).json()
    plan["scenes"] = plan["scenes"][::-1]
    plan["scenes"][0]["included"] = False
    plan["engine"] = "edit"
    saved = client.put(f"/api/projects/{pid}/plan", json=plan).json()
    order = [s["photo_id"] for s in saved["scenes"]]
    changed = client.post(f"/api/projects/{pid}/plan",
                          json={"concept": "잔잔하게", "style": "upbeat", "keep_current_order": True}).json()
    assert changed["style"] == "upbeat" and changed["engine"] == "edit"
    assert [s["photo_id"] for s in changed["scenes"]] == order
    assert [s["included"] for s in changed["scenes"]] == [True, True, True, False]
