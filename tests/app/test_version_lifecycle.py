"""Revision lifecycle, safe persistence, and rendered document-version views."""
import copy
import json
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.components.Adv_doc_version import (
    ChangeTracker, DocumentVersion, PlagiarismEvolutionAnalyzer,
    VersionDiffGenerator, VersionManager, VersionStorageManager,
    integrate_version_control_with_analysis,
    migrate_existing_documents_to_version_control,
)


def history():
    manager = VersionManager()
    for i, text in enumerate(('Original text', 'Original text\nMore research', 'Revised conclusion')):
        version_id = manager.add_version('essay.txt', text, metadata={'author': 'Student'})
        version = manager.get_version('essay.txt', version_id)
        version.timestamp = datetime(2026, 1, 1) + timedelta(days=i)
        version.similarity_score = [0.1, 0.5, 0.9][i]
    return manager


def test_serialization_preserves_unicode_changes_and_metadata():
    version = DocumentVersion('Résumé 文', 'essay.txt', 1)
    version.metadata = {'author': 'Éva'}
    version.change_summary = {'total_added': 2}
    rebuilt = DocumentVersion.from_dict(json.loads(json.dumps(version.to_dict())))
    assert rebuilt.to_dict() == version.to_dict()
    assert rebuilt.size == len('Résumé 文'.encode())
    assert DocumentVersion.decompress(version.compress()) == version.content


def test_delete_keeps_ids_indexes_and_parent_references_consistent():
    manager = history()
    assert manager.delete_version('essay.txt', 2)
    assert manager.get_version('essay.txt', 2) is None
    assert manager.get_version('essay.txt', 3).parent_version == 1
    assert manager.get_current_version('essay.txt').version_id == 3
    assert manager.add_version('essay.txt', 'Original text\nMore research') == 4
    assert manager.get_version_count('essay.txt') == 3
    assert manager.get_all_documents() == ['essay.txt']
    assert manager.get_version_timeline('essay.txt')['Version'].tolist() == ['v1', 'v3', 'v4']
    assert manager.delete_version('essay.txt', 4)
    assert manager.get_current_version('essay.txt').version_id == 3
    for invalid in (0, -1, 999):
        assert manager.get_version('essay.txt', invalid) is None
        assert not manager.delete_version('essay.txt', invalid)
        assert manager.restore_version('essay.txt', invalid) is None


def test_restore_records_a_new_revision_and_selects_it():
    manager = history()
    assert manager.restore_version('essay.txt', 1) == 'Original text'
    current = manager.get_current_version('essay.txt')
    assert current.version_id == 4 and current.parent_version == 1
    assert current.metadata['restored_from'] == 1
    assert manager.add_version('essay.txt', 'Original text') == 4
    assert manager.get_version_count('essay.txt') == 4


def test_import_is_idempotent_and_rebuilds_deduplication_index():
    data = history().export_history('essay.txt')
    restored = VersionManager()
    assert restored.import_history(data)
    assert restored.import_history(data)
    assert restored.export_history('essay.txt') == data
    assert restored.add_version('essay.txt', 'Revised conclusion') == 3
    assert restored.add_version('essay.txt', 'Another revision') == 4


@pytest.mark.parametrize('corruption', ['missing', 'duplicate_id', 'hash', 'parent', 'current', 'next', 'other_document'])
def test_invalid_import_leaves_existing_history_unchanged(corruption):
    manager = history()
    before = manager.export_history('essay.txt')
    data = copy.deepcopy(before)
    if corruption == 'missing': del data['versions'][1]['timestamp']
    if corruption == 'duplicate_id': data['versions'][1]['version_id'] = 1
    if corruption == 'hash': data['versions'][1]['content'] = 'tampered'
    if corruption == 'parent': data['versions'][1]['parent_version'] = 999
    if corruption == 'current': data['current_version'] = 999
    if corruption == 'next': data['next_version_id'] = 1
    if corruption == 'other_document': data['versions'][1]['doc_name'] = 'other.txt'
    assert not manager.import_history(data)
    assert manager.export_history('essay.txt') == before


def test_storage_round_trip_and_invalid_snapshot(tmp_path):
    storage = VersionStorageManager(str(tmp_path))
    manager = history()
    assert storage.load_version_manager('missing.txt') is None
    assert storage.save_version_manager(manager, 'essay.txt')
    assert storage.list_documents() == ['essay.txt']
    restored = storage.load_version_manager('essay.txt')
    assert restored.export_history('essay.txt') == manager.export_history('essay.txt')
    (tmp_path / 'essay.txt.json').write_text('{broken')
    assert storage.load_version_manager('essay.txt') is None
    assert not list(tmp_path.glob('tmp*'))


@pytest.mark.parametrize('name', ['../escape', '..\\escape', '/absolute', 'C:\\outside', '', '..'])
def test_storage_rejects_paths(tmp_path, name):
    storage = VersionStorageManager(str(tmp_path))
    assert not storage.save_version_manager(history(), name)
    assert storage.load_version_manager(name) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('old,new,added,deleted,replaced', [
    ('', '', 0, 0, 0), ('a', 'a\nb', 1, 0, 0), ('a\nb', 'a', 0, 1, 0), ('a', 'b', 0, 0, 1),
])
def test_change_counts_and_diff_outputs(old, new, added, deleted, replaced):
    changes = ChangeTracker().detect_changes(old, new)
    assert (changes['total_added'], changes['total_deleted'], changes['total_replaced']) == (added, deleted, replaced)
    diff = VersionDiffGenerator()
    assert isinstance(diff.generate_html_diff(old, new), str)
    assert bool(diff.generate_unified_diff(old, new)) == (old != new)
    assert bool(diff.generate_context_diff(old, new)) == (old != new)


def test_html_diff_escapes_document_content_and_version_labels():
    result = VersionDiffGenerator().generate_html_diff('<script>alert(1)</script>\nsame', '<img src=x onerror=alert(1)>\nsame', old_ver='<b>1</b>')
    assert '<script>' not in result and '<img ' not in result and '<b>1</b>' not in result
    assert '&lt;script&gt;' in result and '&lt;img ' in result


def test_edit_distance_fallback_is_levenshtein(monkeypatch):
    monkeypatch.setitem(sys.modules, 'nltk', None)
    assert ChangeTracker().detect_changes('kitten', 'sitting')['edit_distance'] == 3


def test_evolution_history_and_plot_have_real_scores():
    manager = history()
    evolution = PlagiarismEvolutionAnalyzer(manager)
    assert evolution.analyze_plagiarism_evolution('missing')['status'] == 'insufficient_data'
    assert evolution.generate_evolution_plot('missing') is None
    analysis = evolution.analyze_plagiarism_evolution('essay.txt')
    assert analysis['trend'] == 'increasing'
    assert len(analysis['significant_changes']) == 2
    assert analysis['metrics']['change_velocity'] == pytest.approx(0.4)
    figure = evolution.generate_evolution_plot('essay.txt')
    assert list(figure.data[0].y) == [0.1, 0.5, 0.9]
    trend = evolution.get_similarity_trend_analysis('essay.txt')
    assert trend['direction'] == 'increasing'
    assert len(trend['forecast']['timestamps']) == 3
    changes = ChangeTracker().get_change_frequency(manager.get_version_history('essay.txt'))
    assert changes['total_changes'] == 2


def test_analysis_integration_and_migration():
    manager = VersionManager()
    report = migrate_existing_documents_to_version_control(manager, [{'filename': 'a', 'content': 'Alpha'}, {'filename': 'empty', 'content': ''}])
    assert report['migrated_documents'] == 1
    result = integrate_version_control_with_analysis(manager, np.array([[0.0, 0.8], [0.8, 0.0]]), ['a', 'b'], {'a': 'Alpha'})
    assert result['a']['similarity_score'] == pytest.approx(0.4)
    assert manager.get_current_version('a').similarity_score == pytest.approx(0.4)
    assert manager.get_current_version('absent') is None
    assert manager.get_version_timeline('absent').empty


@pytest.mark.parametrize('view', ['render_version_control_ui', 'render_plagiarism_evolution_ui', 'render_smart_detection_ui', 'render_global_version_dashboard'])
def test_version_views_render_real_history(view):
    source = '''
from datetime import datetime, timedelta
from app.components.Adv_doc_version import VersionManager, VIEW
manager = VersionManager()
for i in range(3):
    identifier = manager.add_version('essay.txt', 'Original research paragraph. ' * (i + 1))
    version = manager.get_version('essay.txt', identifier)
    version.similarity_score = i * 0.2
    version.timestamp = datetime(2026, 1, 1) + timedelta(days=i)
VIEW(managerARGS)
'''.replace('VIEW', view).replace('ARGS', '' if view == 'render_global_version_dashboard' else ", 'essay.txt'")
    at = AppTest.from_string(source, default_timeout=20).run()
    assert not at.exception, [(e.message, e.stack_trace) for e in at.exception]
    assert at.markdown or at.metric or at.dataframe
