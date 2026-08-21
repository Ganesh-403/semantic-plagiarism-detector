def test_is_user_active():
    """
    Verify that is_user_active correctly handles newly created,
    suspended, and deleted user states.
    """
    # 1. Create a new user (assuming a user creation fixture or helper function exists)
    user = create_test_user(username="test_active_user", email="test@example.com")

    # Verify it returns True for a newly created user
    assert is_user_active(user) is True, (
        "Newly created user should be active by default"
    )

    # 2. Set status to suspended
    set_user_active_status(user.id, "suspended")
    assert is_user_active(user) is False, "Suspended user should not be active"

    # 3. Optional: Set status to deleted or inactive
    set_user_active_status(user.id, "deleted")
    assert is_user_active(user) is False, "Deleted user should not be active"
