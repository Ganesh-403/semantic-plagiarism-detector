"""Administrative account creation and recovery, plus self-service password changes."""
import streamlit as st
from src.db import auth
from app.components.password_management import render_password_change


def render_users_view():
    username = st.session_state.get("username")
    if not st.session_state.get("authenticated") or not username:
        return
    render_password_change()
    if auth.get_user_role(username) != "admin":
        return
    st.subheader("👥 User Management")
    users = auth.get_all_users()
    for user in users:
        st.write(f"User: **{user['username']}** | Role: `{user['role']}`")
    with st.form("create_account", clear_on_submit=True):
        new_username = st.text_input("New account username")
        new_password = st.text_input("Temporary password", type="password")
        role = st.selectbox("New account role", sorted(auth.get_valid_roles()))
        create = st.form_submit_button("Create account")
    if create:
        try:
            auth.add_user(new_username, new_password, role)
            auth.set_password_change_required(new_username, True)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success("Account created. The user must change their temporary password at sign-in.")
    others = [user['username'] for user in users if user['username'] != username]
    if others:
        with st.form("admin_reset_password", clear_on_submit=True):
            target = st.selectbox("Account to reset", others)
            password = st.text_input("Replacement temporary password", type="password")
            reset = st.form_submit_button("Reset account password")
        if reset:
            try:
                auth.update_password(target, password, current_user=username)
                auth.set_password_change_required(target, True)
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
            else:
                st.success("Password reset. Share it privately with the account owner; they must change it at sign-in.")
