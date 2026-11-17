from catalyst.edgar.oddlot import analyse

OFFER = """<p>The Company is offering to purchase up to $50,000,000 of its common stock at a price
not greater than $14.00 nor less than $12.50 per share, net to the seller in cash.</p>
<p>If more Shares are tendered, we will purchase Shares in the following order of priority:
First, all Shares owned in &ldquo;odd lots&rdquo; (less than 100 shares) that have been properly
tendered at or below the Purchase Price; Second, all other Shares on a pro rata basis.</p>"""

PLAIN = "We offer $3.10 per share net to the seller in cash. Shares will be accepted on a pro rata basis."


def test_detects_odd_lot_priority_and_price_range():
    a = analyse(OFFER)
    assert a["odd_lot_priority"] and a["odd_lot_mentions"] == 1
    assert "less than 100 shares" in a["snippet"]
    assert a["price_ranges"] == ["not greater than $14.00 nor less than $12.50"]


def test_plain_tender_is_not_flagged():
    a = analyse(PLAIN)
    assert not a["odd_lot_priority"] and a["odd_lot_mentions"] == 0
    assert a["prices"] == ["$3.10 per share"]
