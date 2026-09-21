from orientation_core.metrics import circular_error_deg


def test_circular_error_wraparound():
    assert circular_error_deg(359, 1) == 2
    assert circular_error_deg(1, 359) == 2
    assert circular_error_deg(90, 270) == 180
