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


def test_locate_keeps_peer_and_industry_pages_when_financial_pages_dominate():
    pages = [
        "营业收入 " * 100,
        "营业收入 " * 90,
        "营业收入 " * 80,
        "同行业可比上市公司 证券简称 中际旭创 新易盛",
        "所属行业 申万一级行业 通信",
    ]

    text, used = pe.locate_relevant_pages(pages, max_pages=3, max_chars=100000)

    assert 3 in used
    assert 4 in used
    assert "中际旭创" in text
    assert "申万一级行业 通信" in text


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


def test_extract_prospectus_fields_normalizes_peer_name_string_and_sw_industry():
    canned = {
        "comparable_company_names": "中际旭创股份有限公司（300308.SZ）、新易盛、天孚通信",
        "sw_level1_industry_name": "通信(申万)",
    }
    res = pe.extract_prospectus_fields(
        b"", extract_json=lambda system, user: canned,
        pages=["同行业可比上市公司 所属行业 申万一级行业 通信"],
    )

    assert res.fields["comparable_company_names"] == ["中际旭创", "新易盛", "天孚通信"]
    assert res.fields["sw_level1_industry_name"] == "通信"
    assert res.fields["sw_level1_industry_code"] == "1000042215000000"


def test_extract_prospectus_fields_maps_csrc_c39_and_keeps_neighbor_peer_names():
    canned = {
        "sw_level1_industry_code": "C39",
        "comparable_company_names": "300308.SZ 中际旭创 300502.SZ 新易盛",
    }
    res = pe.extract_prospectus_fields(
        b"", extract_json=lambda system, user: canned,
        pages=[
            "营业收入 " * 20,
            "同行业可比上市公司",
            "中际旭创（300308.SZ） 新易盛",
            "所属行业 C39 计算机、通信和其他电子设备制造业",
        ],
    )

    assert res.fields["sw_level1_industry_code"] == "1000042193000000"
    assert res.fields["sw_level1_industry_name"] == "电子"
    assert res.fields["comparable_company_names"] == ["中际旭创", "新易盛"]


def test_locate_keeps_neighbor_page_after_peer_anchor():
    pages = [
        "营业收入 " * 20,
        "同行业可比上市公司",
        "中际旭创 新易盛 天孚通信",
    ]

    text, used = pe.locate_relevant_pages(pages, max_pages=3, max_chars=100000)

    assert used == [0, 1, 2]
    assert "中际旭创" in text


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
