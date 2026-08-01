"""Regression coverage for compact cron-failure delivery classification."""

from cron.scheduler import _summarize_cron_failure_for_delivery


JOB = {"id": "job-1", "name": "typed sheet audit", "no_agent": True}


def summarize(error: str) -> str:
    return _summarize_cron_failure_for_delivery(JOB, error)


def test_bare_business_row_403_is_not_misclassified_as_provider_auth():
    result = summarize(
        "RuntimeError: affiliate_sheet_typed_value_repair_required: "
        "vertical_value_not_allowed at sheet row 403"
    )

    assert "provider authentication error" not in result
    assert "sheet row 403" in result


def test_traceback_summary_uses_final_exception_instead_of_header():
    result = summarize(
        "Traceback (most recent call last):\n"
        "  File \"affiliate_approval_auto_reconciler.py\", line 788, in prepare_sheet_schema\n"
        "RuntimeError: affiliate_sheet_typed_value_repair_required at sheet row 403\n"
    )

    assert "provider authentication error" not in result
    assert "affiliate_sheet_typed_value_repair_required" in result
    assert "Traceback" not in result


def test_bare_business_record_429_is_not_misclassified_as_rate_limit():
    result = summarize("RuntimeError: invalid marketplace record 429")

    assert "provider rate limit" not in result
    assert "record 429" in result


def test_successful_authenticated_wording_is_not_an_auth_failure():
    result = summarize("RuntimeError: authenticated provider readback PASS at row 403")

    assert "provider authentication error" not in result


def test_contextual_http_403_is_classified_as_provider_auth():
    result = summarize("HTTP 403 Forbidden from upstream provider")

    assert "provider authentication error" in result


def test_contextual_error_code_401_is_classified_as_provider_auth():
    result = summarize("Error code: 401 - invalid credentials")

    assert "provider authentication error" in result


def test_explicit_authentication_failure_is_classified_as_provider_auth():
    result = summarize("Authentication failed while refreshing access token")

    assert "provider authentication error" in result


def test_contextual_http_429_is_classified_as_rate_limit():
    result = summarize("HTTP 429 Too Many Requests")

    assert "provider rate limit" in result
