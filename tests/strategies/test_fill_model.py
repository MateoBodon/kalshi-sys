from kalshi_alpha.strategies.index.fill_model import estimate_maker_fill_prob


def test_fill_prob_higher_near_mid():
    near = estimate_maker_fill_prob(distance_to_mid_cents=1.0, time_to_expiry_minutes=30, spread_cents=6.0)
    far = estimate_maker_fill_prob(distance_to_mid_cents=12.0, time_to_expiry_minutes=30, spread_cents=6.0)
    assert near > far


def test_fill_prob_grows_with_time():
    short = estimate_maker_fill_prob(distance_to_mid_cents=3.0, time_to_expiry_minutes=2, spread_cents=8.0)
    long = estimate_maker_fill_prob(distance_to_mid_cents=3.0, time_to_expiry_minutes=45, spread_cents=8.0)
    assert long > short


def test_fill_prob_clamped_between_zero_and_one():
    prob = estimate_maker_fill_prob(distance_to_mid_cents=0.0, time_to_expiry_minutes=120, spread_cents=2.0)
    assert 0.0 <= prob <= 1.0
