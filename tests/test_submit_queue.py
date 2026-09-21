from manual_submit import build_queue


def test_queue_reads_only_top_level_urls_and_skips_sitemaps():
    state = {
        "urls": [
            "https://example.com/a",
            "https://example.com/sitemap.xml",
            "https://example.com/b",
        ],
        "completed": ["https://example.com/a"],
        "skipped": [],
        "index": 0,
        "root_sitemaps": ["https://example.com/sitemap.xml"],
    }

    all_urls, pending, _ = build_queue(state)
    assert all_urls == ["https://example.com/a", "https://example.com/b"]
    assert pending == ["https://example.com/b"]
