from src.wait_kernel import parse_status, wait


def test_parse_status_formats():
    assert parse_status('u/k has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert parse_status('u/k has status "running"') == "running"
    assert parse_status("403 Client Error: Forbidden") is None  # transient CLI noise


def test_wait_ignores_stale_complete_until_run_starts():
    seq = iter(["complete", None, "queued", "running", "complete"])
    assert wait(lambda: next(seq), sleep=lambda s: None, max_polls=10) == "complete"


def test_wait_reports_error_and_timeout():
    assert wait(lambda: "error", sleep=lambda s: None, max_polls=3) == "error"
    assert wait(lambda: "running", sleep=lambda s: None, max_polls=3) == "timeout"
