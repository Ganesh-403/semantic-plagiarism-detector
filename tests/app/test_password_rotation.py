from src.db import auth


def test_required_rotation_blocks_dashboard_until_password_changed(upload_dashboard):
    at,_ = upload_dashboard
    auth.set_password_change_required("upload_admin",True)
    at.run()
    assert not at.exception
    assert any("before continuing" in item.value for item in at.warning)
    assert not at.metric
    next(w for w in at.text_input if w.label == "Current password").set_value("Upload-Test-Password!984")
    next(w for w in at.text_input if w.label == "New password").set_value("Rotated-Upload-Password!985")
    next(w for w in at.text_input if w.label == "Confirm new password").set_value("Rotated-Upload-Password!985")
    next(b for b in at.button if b.label == "Change password").click().run()
    assert not at.exception, [(e.message,e.stack_trace) for e in at.exception]
    assert not at.session_state["authenticated"]
    result = auth.verify_user("upload_admin","Rotated-Upload-Password!985",return_details=True)
    assert result["authenticated"] and not result["must_change_password"] and not result["password_expired"]


def test_changed_password_invalidates_an_open_dashboard_session(upload_dashboard):
    at,_ = upload_dashboard
    auth.update_password("upload_admin","Rotated-Upload-Password!985")
    at.run()
    assert not at.exception
    assert not at.session_state["authenticated"]


def test_suspended_account_invalidates_an_open_dashboard_session(upload_dashboard):
    at, _ = upload_dashboard
    auth.set_user_active_status("upload_admin", False)
    at.run()
    assert not at.exception
    assert not at.session_state["authenticated"]


def test_deleted_account_invalidates_an_open_dashboard_session(upload_dashboard):
    at, _ = upload_dashboard
    auth.delete_user("upload_admin")
    at.run()
    assert not at.exception
    assert not at.session_state["authenticated"]
