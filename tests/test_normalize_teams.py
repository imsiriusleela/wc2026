from wcpredictor.data.normalize_teams import canonical


def test_known_alias():
    assert canonical("Korea Republic") == "South Korea"
    assert canonical("IR Iran") == "Iran"
    assert canonical("Côte d'Ivoire") == "Ivory Coast"
    assert canonical("USA") == "United States"


def test_unknown_passthrough():
    assert canonical("Brazil") == "Brazil"
    assert canonical("Germany") == "Germany"
    assert canonical("SomeFantasyFC") == "SomeFantasyFC"


def test_idempotent():
    for name in ("South Korea", "Iran", "Brazil", "Germany"):
        assert canonical(canonical(name)) == canonical(name)
