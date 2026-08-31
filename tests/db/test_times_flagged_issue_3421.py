"""
tests/db/test_times_flagged_issue_3421.py
-------------------------------------------
Tests for Issue #3421: recurring incidents should update last_seen and
increment a times_flagged counter instead of just overwriting the row.
"""

from datetime import datetime, timedelta, timezone

from src.db.incidents import sync_flagged_incidents
from src.db.migrations.corpus import CORPUS_MIGRATIONS, CORPUS_SCHEMA_VERSION


def _flag(doc_a: str, doc_b: str, similarity: float) -> dict:
    return {"doc_a": doc_a, "doc_b": doc_b, "similarity": similarity}


class TestTimesFlaggedMigration:
    """Verify the times_flagged column is registered as a migration."""

    def test_migration_019_registered(self):
        assert 19 in CORPUS_MIGRATIONS
        assert CORPUS_MIGRATIONS[19].__name__ == "migration_019_add_times_flagged"

    def test_schema_version_bumped(self):
        assert CORPUS_SCHEMA_VERSION >= 19

    def test_migrations_are_sequential_with_no_gaps(self):
        """Regression guard: this migration must not collide with, or leave
        a gap after, migrations added by other contributors in the same
        version range (17, 18 were taken by unrelated work between when
        this fix was first written and when it was re-applied)."""
        assert list(CORPUS_MIGRATIONS.keys()) == list(
            range(1, CORPUS_SCHEMA_VERSION + 1)
        )


class TestTimesFlaggedIncrement:
    """Verify sync_flagged_incidents increments times_flagged on re-flag."""

    def test_new_incident_defaults_to_one(self, mock_db):
        """A document pair flagged for the first time starts at times_flagged=1."""
        results = sync_flagged_incidents(
            [_flag("essay_a.pdf", "essay_b.pdf", 0.85)], mock_db
        )

        assert len(results) == 1
        assert results[0].times_flagged == 1

    def test_reflagging_same_pair_increments_counter(self, mock_db):
        """Re-flagging the same document pair in a later scan increments
        times_flagged instead of resetting it."""
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        first_scan = (base_time).isoformat()
        sync_flagged_incidents(
            [_flag("essay_a.pdf", "essay_b.pdf", 0.85)],
            mock_db,
            now=first_scan,
        )

        second_scan = (base_time + timedelta(days=1)).isoformat()
        results = sync_flagged_incidents(
            [_flag("essay_a.pdf", "essay_b.pdf", 0.88)],
            mock_db,
            now=second_scan,
        )

        assert len(results) == 1
        assert results[0].times_flagged == 2

    def test_reflagging_updates_last_seen(self, mock_db):
        """Re-flagging updates last_seen to the most recent scan timestamp."""
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        sync_flagged_incidents(
            [_flag("essay_a.pdf", "essay_b.pdf", 0.85)],
            mock_db,
            now=base_time.isoformat(),
        )

        second_scan = (base_time + timedelta(days=2)).isoformat()
        results = sync_flagged_incidents(
            [_flag("essay_a.pdf", "essay_b.pdf", 0.90)],
            mock_db,
            now=second_scan,
        )

        assert results[0].last_seen == second_scan

    def test_reflagging_three_times_increments_to_three(self, mock_db):
        """The counter keeps incrementing across more than two re-flags."""
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for i in range(3):
            timestamp = (base_time + timedelta(days=i)).isoformat()
            results = sync_flagged_incidents(
                [_flag("essay_a.pdf", "essay_b.pdf", 0.80 + i * 0.02)],
                mock_db,
                now=timestamp,
            )

        assert results[0].times_flagged == 3

    def test_unrelated_pair_unaffected_by_reflagging(self, mock_db):
        """Re-flagging one pair must not bump the counter for a different,
        unrelated document pair."""
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        sync_flagged_incidents(
            [
                _flag("essay_a.pdf", "essay_b.pdf", 0.85),
                _flag("other_x.pdf", "other_y.pdf", 0.70),
            ],
            mock_db,
            now=base_time.isoformat(),
        )

        second_scan = (base_time + timedelta(days=1)).isoformat()
        results = sync_flagged_incidents(
            [_flag("essay_a.pdf", "essay_b.pdf", 0.88)],
            mock_db,
            now=second_scan,
        )

        by_pair = {(r.document_a, r.document_b): r for r in results}
        assert by_pair[("essay_a.pdf", "essay_b.pdf")].times_flagged == 2
        assert by_pair[("other_x.pdf", "other_y.pdf")].times_flagged == 1

    def test_reflagging_pair_regardless_of_argument_order(self, mock_db):
        """The pair (A, B) and (B, A) resolve to the same incident, so
        flagging it in the opposite order still increments the same
        counter rather than creating a second incident."""
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        sync_flagged_incidents(
            [_flag("essay_a.pdf", "essay_b.pdf", 0.85)],
            mock_db,
            now=base_time.isoformat(),
        )

        second_scan = (base_time + timedelta(days=1)).isoformat()
        results = sync_flagged_incidents(
            [_flag("essay_b.pdf", "essay_a.pdf", 0.86)],  # order swapped
            mock_db,
            now=second_scan,
        )

        assert len(results) == 1
        assert results[0].times_flagged == 2

    def test_reflagging_preserves_a_dismissed_review_status(self, mock_db):
        """A pair that was previously marked 'Dismissed' by a reviewer must
        stay Dismissed when re-flagged -- the ON CONFLICT clause updates
        similarity_score/severity_rank/last_seen/times_flagged only, never
        review_status, so a reviewer's decision isn't silently overwritten
        by the next scan."""
        from src.db.incidents import _get_connection, build_incident_id

        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sync_flagged_incidents(
            [_flag("essay_a.pdf", "essay_b.pdf", 0.85)],
            mock_db,
            now=base_time.isoformat(),
        )

        incident_id = build_incident_id("essay_a.pdf", "essay_b.pdf")
        with _get_connection(mock_db) as conn:
            conn.execute(
                "UPDATE plagiarism_incidents SET review_status = 'Dismissed' "
                "WHERE incident_id = ?",
                (incident_id,),
            )
            conn.commit()

        second_scan = (base_time + timedelta(days=1)).isoformat()
        results = sync_flagged_incidents(
            [_flag("essay_a.pdf", "essay_b.pdf", 0.90)],
            mock_db,
            now=second_scan,
        )

        assert results[0].times_flagged == 2
        assert results[0].review_status == "Dismissed"
