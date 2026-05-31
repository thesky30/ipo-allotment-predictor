import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ui_helpers


def test_manual_form_sections_include_comparable_pe_and_source_labels():
    sections = ui_helpers.manual_form_sections()

    assert "财务估值" in sections
    assert "comparable_pe_avg_ex_nonrecurring" in sections["财务估值"]
    assert ui_helpers.prefill_source_label("comparable_pe_avg_ex_nonrecurring", "peer_pe") == "Tushare peer PE参考"
    assert ui_helpers.prefill_source_label("latest_revenue_100m_yuan", "prospectus") == "招股书"
