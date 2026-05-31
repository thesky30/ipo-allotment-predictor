import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import prospectus_extract as pe

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_locate_picks_anchor_pages_and_caps():
    pages = [
        "目录 第一节 释义 ......无关内容" * 50,
        "第八节 财务会计信息 营业收入 营业收入 扣除非经常性损益" * 5,
        "无关页" * 50,
        "募集资金运用 本次拟募集资金 可比公司 同行业可比上市公司",
    ]
    text, used = pe.locate_relevant_pages(pages, max_pages=2, max_chars=100000)
    assert used == [1, 3]
    assert "营业收入" in text and "募集资金" in text
    assert 0 not in used and 2 not in used


def test_locate_respects_char_cap():
    pages = ["营业收入 " * 1000, "募集资金 " * 1000]
    text, used = pe.locate_relevant_pages(pages, max_pages=8, max_chars=500)
    assert len(text) <= 500
    assert used


def test_extract_prospectus_fields_with_injected_llm():
    canned = {
        "latest_revenue_100m_yuan": 7.2,
        "revenue_cagr_3y_pct": 25.0,
        "comparable_pe_avg_ex_nonrecurring": 41.3,
        "comparable_company_names": ["中际旭创", "新易盛"],
        "expected_fundraising_100m_yuan": 11.0,
        "board": "科创板",
    }
    res = pe.extract_prospectus_fields(
        b"", extract_json=lambda system, user: canned,
        pages=["营业收入 募集资金 可比公司"],
    )
    assert res.fields["latest_revenue_100m_yuan"] == 7.2
    assert "comparable_pe_avg_ex_nonrecurring" not in res.fields
    assert res.fields["comparable_company_names"] == ["中际旭创", "新易盛"]
    assert "board" not in res.fields


@pytest.mark.skipif(
    not list(FIXTURES.glob("*招股*.pdf")),
    reason="把真实招股书 PDF（文件名含『招股』）放进 tests/fixtures/ 才跑此测试",
)
def test_extract_text_pages_real_prospectus():
    pdf = next(FIXTURES.glob("*招股*.pdf"))
    pages = pe.extract_pages(pdf.read_bytes())
    assert len(pages) > 20
    text, used = pe.locate_relevant_pages(pages)
    assert used and len(text) > 200
