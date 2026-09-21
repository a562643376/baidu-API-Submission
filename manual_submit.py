from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_STATE = Path("data/baidu-manual-submit-state.json")
DEFAULT_COOKIES = Path("data/cookies.json")
DEFAULT_TARGET = "https://ziyuan.baidu.com/linksubmit/url"


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize_urls(values):
    if not isinstance(values, list):
        return []
    out = []
    seen = set()
    for item in values:
        if not isinstance(item, str):
            continue
        url = item.strip()
        if not url or url in seen:
            continue
        try:
            parsed = urlparse(url)
        except Exception:
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if "sitemap" in parsed.path.lower() and parsed.path.lower().endswith(".xml"):
            continue
        seen.add(url)
        out.append(url)
    return out


def build_queue(state):
    urls = normalize_urls(state.get("urls", []))
    completed = set(normalize_urls(state.get("completed", [])))
    skipped = set(normalize_urls(state.get("skipped", [])))

    current = state.get("current_url", "")
    try:
        index = int(state.get("index", 0))
    except Exception:
        index = 0
    index = max(0, min(index, len(urls)))

    if current in urls and current not in completed and current not in skipped:
        index = urls.index(current)

    pending = [u for u in urls[index:] if u not in completed and u not in skipped]
    return urls, pending, index


def load_playwright_cookies(path: Path):
    raw = load_json(path)
    values = raw.get("cookies", raw) if isinstance(raw, dict) else raw
    if not isinstance(values, list):
        return []

    out = []
    for c in values:
        if not isinstance(c, dict) or not c.get("name") or "value" not in c:
            continue
        item = {"name": str(c["name"]), "value": str(c["value"])}
        if c.get("domain"):
            item["domain"] = c["domain"]
            item["path"] = c.get("path", "/")
        elif c.get("url"):
            item["url"] = c["url"]
        else:
            continue

        if c.get("expires") not in (None, -1):
            try:
                item["expires"] = float(c["expires"])
            except Exception:
                pass
        if "httpOnly" in c:
            item["httpOnly"] = bool(c["httpOnly"])
        if "secure" in c:
            item["secure"] = bool(c["secure"])
        if c.get("sameSite") in {"Strict", "Lax", "None"}:
            item["sameSite"] = c["sameSite"]
        out.append(item)
    return out


def is_visible(locator):
    try:
        return locator.count() > 0 and locator.first.is_visible()
    except Exception:
        return False


def detect(page):
    success = page.locator("#dialog-content .content-div").filter(has_text="链接提交成功")
    if is_visible(success):
        return "success"
    if is_visible(page.locator("div.passMod_dialog-container")):
        return "captcha"
    if is_visible(page.locator("#dialog")):
        return "dialog"
    return "none"


def wait_result(page, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = detect(page)
        if state != "none":
            return state
        page.wait_for_timeout(150)
    return "none"


def wait_manual(page, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if detect(page) == "success":
            return True
        page.wait_for_timeout(200)
    return False


def advance_state(state_path: Path, state: dict, url: str, all_urls: list[str]):
    completed = normalize_urls(state.get("completed", []))
    skipped = normalize_urls(state.get("skipped", []))
    if url not in completed:
        completed.append(url)
    skipped = [x for x in skipped if x != url]

    state["completed"] = completed
    state["skipped"] = skipped

    pos = all_urls.index(url)
    done = set(completed) | set(skipped)
    next_index = pos + 1
    while next_index < len(all_urls) and all_urls[next_index] in done:
        next_index += 1

    state["index"] = next_index
    state["current_url"] = all_urls[next_index] if next_index < len(all_urls) else ""
    state["status"] = "running" if next_index < len(all_urls) else "completed"
    state["updated_at"] = now_iso()
    save_json(state_path, state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--cookies", default=str(DEFAULT_COOKIES))
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--manual-timeout", type=int, default=300)
    ap.add_argument("--initial-timeout", type=int, default=30)
    ap.add_argument("--delay-ms", type=int, default=800)
    args = ap.parse_args()

    state_path = Path(args.state)
    cookies_path = Path(args.cookies)

    if not state_path.exists():
        raise SystemExit(f"Missing state file: {state_path}")
    if not cookies_path.exists():
        raise SystemExit(f"Missing cookies file: {cookies_path}")

    state = load_json(state_path)
    all_urls, pending, start_index = build_queue(state)
    print(f"top_level_urls={len(all_urls)} start_index={start_index} pending={len(pending)}")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=False)
        context = browser.new_context(locale="zh-CN")
        context.add_cookies(load_playwright_cookies(cookies_path))
        page = context.new_page()
        page.goto(args.target, wait_until="domcontentloaded", timeout=60000)
        page.locator("#url").wait_for(state="visible", timeout=30000)

        for i, url in enumerate(pending, start=1):
            print(f"[{i}/{len(pending)}] {url}")
            page.locator("#url").fill(url)
            page.locator("#saveBtn").click()

            result = wait_result(page, args.initial_timeout)
            ok = result == "success"

            if result == "captcha":
                print("检测到人机验证，请在浏览器中人工完成。")
                ok = wait_manual(page, args.manual_timeout)

            if not ok:
                state["status"] = "paused"
                state["current_url"] = url
                state["updated_at"] = now_iso()
                save_json(state_path, state)
                print("当前 URL 未确认成功，断点已保存。")
                break

            advance_state(state_path, state, url, all_urls)

            btn = page.locator("#dialog-foot button.btn-big-blue")
            if is_visible(btn):
                btn.first.click()

            page.wait_for_timeout(args.delay_ms)

        browser.close()


if __name__ == "__main__":
    main()
