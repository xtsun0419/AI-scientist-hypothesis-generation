from lit_agent.text import doi_url, normalize_doi, normalize_title


def test_normalize_doi_from_url_and_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1016/J.JMMM.2020.166970") == "10.1016/j.jmmm.2020.166970"
    assert normalize_doi("doi:10.1109/TMAG.2019.12345.") == "10.1109/tmag.2019.12345"
    assert doi_url("10.1016/j.jmmm.2020.166970") == "https://doi.org/10.1016/j.jmmm.2020.166970"


def test_normalize_title() -> None:
    assert normalize_title("  Coercivity in Nd-Fe-B Permanent Magnets! ") == "coercivity in nd fe b permanent magnets"
