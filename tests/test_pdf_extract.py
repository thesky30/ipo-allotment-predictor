import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import pdf_extract

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_extract_ipo_fields_with_injected_llm():
    canned = {
        "board": "科创板",
        "subscription_deadline_date": "2025-03-18",
        "lead_underwriter": "中信证券股份有限公司",
        "total_issue_shares_10k": 4000,
        "industry_pe_at_ipo": 38.5,
        "not_a_field": "ignored",
        "expected_fundraising_100m_yuan": None,
    }
    res = pdf_extract.extract_ipo_fields(
        b"", extract_json=lambda system, user: canned, text="一些公告文本"
    )
    assert res.fields["board"] == "科创板"
    assert res.fields["total_issue_shares_10k"] == 4000
    assert "not_a_field" not in res.fields
    assert "expected_fundraising_100m_yuan" not in res.fields
    assert res.text_chars == len("一些公告文本")


def test_schema_keys_match_assemble_contract():
    import feature_assembly  # noqa: F401
    assert "subscription_deadline_date" in pdf_extract.FIELD_SCHEMA
    assert "lead_underwriter" in pdf_extract.FIELD_SCHEMA
    assert "sw_level1_industry_code" in pdf_extract.FIELD_SCHEMA


@pytest.mark.skipif(
    not list(FIXTURES.glob("*.pdf")),
    reason="drop a real 巨潮 公告 PDF into tests/fixtures/ to run the real-PDF text test",
)
def test_extract_text_reads_real_pdf():
    pdf = next(FIXTURES.glob("*.pdf"))
    text = pdf_extract.extract_text(pdf.read_bytes())
    assert len(text) > 200
