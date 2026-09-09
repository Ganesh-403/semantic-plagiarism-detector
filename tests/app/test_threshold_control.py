from streamlit.testing.v1 import AppTest


def test_presets_and_manual_slider_changes_survive_reruns():
    at = AppTest.from_string("from app.components.threshold_control import render_threshold_control; render_threshold_control()").run()
    assert not at.exception
    at.radio[0].set_value("Custom").run()
    assert not at.exception
    at.slider[0].set_value(.72).run()
    assert not at.exception
    assert at.radio[0].value == "Custom"
    assert at.slider[0].value == .72
    at.radio[0].set_value("Strict (0.80)").run()
    assert not at.exception
    assert at.slider[0].value == .80
    at.slider[0].set_value(.63).run()
    assert not at.exception
    assert at.radio[0].value == "Custom"
    at.run()
    assert at.slider[0].value == .63
