"""
Stopword Manager UI Component

Provides UI for managing stopword lists in the plagiarism detector.
Users can view, add, remove, and toggle stopword lists.
"""

import streamlit as st

from src.core.stopwords import get_stopword_manager


def render_stopword_manager_ui() -> None:
    """
    Render the stopword manager UI in Streamlit.
    """
    manager = get_stopword_manager()

    st.markdown("### 🛑 Stopword Manager")
    st.caption("Manage stopwords used for lexical similarity filtering.")

    # Statistics
    stats = manager.get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🇬🇧 English", stats["english"])
    col2.metric("📚 Academic", stats["academic"])
    col3.metric("✏️ Custom", stats["custom"])
    col4.metric("📊 Total", stats["total"])

    st.divider()

    # Toggle stopword lists
    st.markdown("### 🔘 Toggle Stopword Lists")
    col1, col2, col3 = st.columns(3)

    with col1:
        english_enabled = st.checkbox(
            "🇬🇧 English Stopwords",
            value=manager._enabled.get("english", True),
            key="toggle_english_stopwords",
            help="Common English stopwords (a, the, and, etc.)",
        )
        if english_enabled != manager._enabled.get("english", True):
            if english_enabled:
                manager.enable_list("english")
            else:
                manager.disable_list("english")
            st.rerun()

    with col2:
        academic_enabled = st.checkbox(
            "📚 Academic Stopwords",
            value=manager._enabled.get("academic", True),
            key="toggle_academic_stopwords",
            help="Domain-specific academic stopwords (figure, table, etc.)",
        )
        if academic_enabled != manager._enabled.get("academic", True):
            if academic_enabled:
                manager.enable_list("academic")
            else:
                manager.disable_list("academic")
            st.rerun()

    with col3:
        custom_enabled = st.checkbox(
            "✏️ Custom Stopwords",
            value=manager._enabled.get("custom", True),
            key="toggle_custom_stopwords",
            help="User-added custom stopwords",
        )
        if custom_enabled != manager._enabled.get("custom", True):
            if custom_enabled:
                manager.enable_list("custom")
            else:
                manager.disable_list("custom")
            st.rerun()

    st.divider()

    # Add custom stopwords
    st.markdown("### ➕ Add Custom Stopwords")

    col1, col2 = st.columns([3, 1])
    with col1:
        new_stopwords = st.text_area(
            "Enter stopwords (one per line)",
            placeholder="e.g.\nfig\netc\nibid",
            key="custom_stopwords_input",
            height=100,
            help="Add custom stopwords to filter out from lexical matching.",
        )
    with col2:
        st.write("")
        if st.button(
            "➕ Add Stopwords", key="add_stopwords_btn", use_container_width=True
        ):
            if new_stopwords.strip():
                words = [w.strip() for w in new_stopwords.split("\n") if w.strip()]
                added = manager.add_custom_list(words)
                if added > 0:
                    st.success(f"✅ Added {added} custom stopwords!")
                    st.rerun()
                else:
                    st.warning("No new stopwords added.")
            else:
                st.warning("Please enter at least one stopword.")

    st.divider()

    # Manage custom stopwords
    st.markdown("### 📋 Custom Stopwords List")

    if manager.custom:
        # Display current custom stopwords
        custom_list = sorted(manager.custom)

        # Pagination
        items_per_page = 20
        total_pages = (len(custom_list) + items_per_page - 1) // items_per_page

        page = st.session_state.get("stopword_page", 1)
        if page > total_pages:
            page = 1
            st.session_state["stopword_page"] = page

        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(custom_list))

        st.caption(
            f"Showing {start_idx + 1}-{end_idx} of {len(custom_list)} custom stopwords"
        )

        # Display with remove buttons
        for idx, word in enumerate(custom_list[start_idx:end_idx]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.code(word, language=None)
            with col2:
                if st.button("🗑️", key=f"remove_custom_{word}_{idx}"):
                    if manager.remove_stopword(word):
                        st.success(f"Removed '{word}'")
                        st.rerun()

        # Pagination controls
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀️ Previous", key="stopword_prev", disabled=(page <= 1)):
                    st.session_state["stopword_page"] = page - 1
                    st.rerun()
            with col2:
                st.caption(f"Page {page} of {total_pages}")
            with col3:
                if st.button(
                    "Next ▶️", key="stopword_next", disabled=(page >= total_pages)
                ):
                    st.session_state["stopword_page"] = page + 1
                    st.rerun()

        # Clear all custom stopwords
        if st.button("🗑️ Clear All Custom Stopwords", key="clear_custom_btn"):
            manager.clear_custom()
            st.success("All custom stopwords cleared!")
            st.rerun()

    else:
        st.info("No custom stopwords added yet. Add some using the form above.")

    st.divider()

    # View all stopwords
    with st.expander("👁️ View All Stopwords", expanded=False):
        all_stopwords = sorted(manager.get_stopwords())

        search = st.text_input("🔍 Search stopwords", placeholder="Search...")

        if search:
            filtered = [w for w in all_stopwords if search.lower() in w.lower()]
            st.caption(f"Found {len(filtered)} matching stopwords")
            st.code(
                "\n".join(filtered[:100]) + ("\n..." if len(filtered) > 100 else ""),
                language=None,
            )
        else:
            st.caption(f"Total: {len(all_stopwords)} stopwords")
            st.code(
                "\n".join(all_stopwords[:100])
                + ("\n..." if len(all_stopwords) > 100 else ""),
                language=None,
            )

    st.divider()

    # Reset to defaults
    if st.button("🔄 Reset to Default Stopwords", key="reset_stopwords_btn"):
        manager.custom.clear()
        manager._enabled = {
            "english": True,
            "academic": True,
            "custom": True,
        }
        manager._clear_cache()
        manager.save_custom_stopwords()
        st.success("✅ Reset to default stopwords!")
        st.rerun()


def render_stopword_status_badge() -> str:
    """Render a badge showing stopword status."""
    manager = get_stopword_manager()
    stats = manager.get_stats()
    enabled_count = sum(1 for v in stats["enabled"].values() if v)
    return f"🛑 Stopwords: {stats['total']} ({enabled_count}/3 lists enabled)"
