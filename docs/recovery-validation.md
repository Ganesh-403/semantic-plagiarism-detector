# Recovery validation

The complete suite remains red. This report records the local Windows/Python 3.13
run; it is not a substitute for GitHub's Linux runners or a deployment check.

## Complete suite

Command:

```console
python -m pytest -n 2 --dist=loadscope --tb=short -q --cov=src --cov=app --cov-report=xml --junitxml=suite.xml
```

Result: **677 failed, 8982 passed, 25 skipped, 1 xfailed, 269 warnings, 25 errors in 942.93s (0:15:42)**.

Line coverage: **52.26%**, below the existing 85% gate. The 90% changed-line
coverage gate is also retained. Existing legacy exclusions remain unchanged;
no new skips, expected failures or lower coverage thresholds hide this result.

This broad run overlapped final authentication regression additions, a test assertion
correction, and pre-commit's temporary restoration of unstaged files. Those final
changes were checked separately below. Shared legacy test state may affect
individual outcomes; reproduce the broad suite in isolation when investigating.

## Passing independent and targeted checks

- Streamlit login, real two-document embedding analysis, result rendering, corpus
  persistence and rerun without duplicates: `scripts/smoke_streamlit.py` with
  `SMOKE_REAL_MODEL=true`, `SEMANTIC_PLAGIARISM_MODEL=all-MiniLM-L6-v2` and
  `CROSS_ENCODER_RERANKING_ENABLED=false`. File bytes are injected at the upload
  boundary; the model, parser, FAISS and application views run normally.
- Fresh API startup, password login, signed-token corpus read and private-route
  rejection: `python scripts/smoke_api.py`.
- Focused backlog/API/localization checks plus syntax/conflict guards: 1,491 passed
  in the final grouped run; the one test assertion correction above passed its
  subsequent targeted rerun. That group contains 263 backlog/API/localization tests and 1,229
  additional startup, syntax and conflict checks.
- All 610 source files parse using Python 3.11 grammar; Ruff and YAML lint pass.
- Pre-commit checks, including secret detection, pass across the changed files
  after narrowly marking synthetic test credentials. No secret hashes were added
  to the existing baseline; one line reference was refreshed.
- Real upload/cleanup memory check: 100 iterations of 1.2 MB each, 0.00 MiB retained
  allocation delta against the 10 MiB limit.

## External checks

GitHub reported `action_required` for all seven workflows on the initial PR head.
Maintainer action is required before those runners execute. Docker and the upstream
Streamlit Community Cloud site have not been built/deployed from this Windows host.

The dependency audit still reports deep-translator PYSEC-2022-252 and NLTK
PYSEC-2026-3740 without fixed versions listed. The runtime's PyMuPDF/EbookLib licenses
also conflict with the repository's existing blanket copyleft policy. Findings and
policy have not been suppressed. See [deployment and recovery notes](project-recovery.md).

## Modules with failing or erroring test cases

Counts below identify work to investigate; they do not classify every failure as
an implementation defect or as a stale test. Some cases have both a failure and a
teardown error, so module totals need not equal the separate summary categories.

| Test module | Cases |
| --- | ---: |
| `tests.api.test_heatmap` | 36 |
| `tests.api.test_app` | 20 |
| `tests.db.test_translation_cache` | 19 |
| `tests.scripts.test_verify_structure` | 19 |
| `tests.core.test_document_parser` | 14 |
| `tests.utils.test_sso` | 13 |
| `tests.api.test_document_versions` | 13 |
| `tests.core.test_batch_processor` | 12 |
| `tests.visualization.test_network_graph` | 11 |
| `tests.core.test_text_chunking` | 11 |
| `tests.app.test_system_health_ui` | 11 |
| `tests.cli.test_cli` | 9 |
| `tests.app.test_backup_daemon_race_condition` | 9 |
| `tests.utils.test_migrate_redis_keys` | 9 |
| `tests.core.test_cross_lingual` | 8 |
| `tests.core.test_teext_normalization_properties` | 8 |
| `tests.core.test_text_normalization_properties` | 8 |
| `tests.scripts.test_generate_seed_data` | 8 |
| `tests.scripts.test_benchmark_chunking` | 8 |
| `tests.visualization.test_analytics` | 7 |
| `tests.api.test_api` | 7 |
| `tests.utils.test_file_validator_signatures` | 7 |
| `tests.visualization.test_font_scale` | 7 |
| `tests.utils.test_pdf_report` | 6 |
| `tests.utils.test_file_validator` | 6 |
| `tests.core.test_text_chunking_min_words` | 6 |
| `tests.api.test_middleware` | 6 |
| `tests.test_cross_lingual` | 6 |
| `tests.utils.test_trend_report_exporter` | 6 |
| `tests.security.test_ssrf_protector` | 5 |
| `tests.core.test_metrics` | 5 |
| `tests.app.test_faiss_results` | 5 |
| `tests.utils.test_sso_config_error_issue_2583` | 5 |
| `tests.test_hybrid_analyzer.TestSimilarityMetrics` | 5 |
| `tests.db.test_corpus_db_connections` | 5 |
| `tests.core.test_synchronization` | 5 |
| `tests.api.test_otel_middleware` | 5 |
| `tests.api.test_password_reset` | 5 |
| `tests.infrastructure.test_pdf_fixtures` | 5 |
| `tests.utils.test_windows_reserved_names_issue_3725` | 5 |
| `tests.db.test_auth_expiration` | 5 |
| `tests.utils.test_redis_cache` | 4 |
| `tests.core.test_processing_module_syntax_issue_4189` | 4 |
| `tests.core.test_document_parser_ocr` | 4 |
| `tests.db.migrations` | 4 |
| `tests.integration.test_task_queue` | 4 |
| `tests.app.test_copy_snippet` | 4 |
| `tests.api.test_endpoints` | 4 |
| `tests.core.test_semantic_alignment` | 4 |
| `tests.test_hybrid_analyzer.TestHybridAnalyzer` | 4 |
| `tests.db.test_annotations_db` | 4 |
| `tests.api.test_health_probes_issue_2923` | 4 |
| `tests.api.test_upload_duplicate` | 4 |
| `tests.app.test_notification_preferences_ui` | 4 |
| `tests.core.test_cross_lingual_detector` | 4 |
| `tests.core.test_sql_scan` | 4 |
| `tests.app.test_theme` | 3 |
| `tests.core.test_faiss_index` | 3 |
| `tests.security.test_mime_validator` | 3 |
| `tests.utils.test_badge_generator_importable_issue_3847` | 3 |
| `tests.core.test_lexical_similarity` | 3 |
| `tests.core.test_app_config` | 3 |
| `tests.core.test_translator` | 3 |
| `tests.core.test_internet_engine` | 3 |
| `tests.utils.test_coleman_liau_readability_issue_3704` | 3 |
| `tests.visualization.test_lazy_analytics` | 3 |
| `tests.api.test_error_response_schemas` | 3 |
| `tests.cli.test_cli_purge` | 3 |
| `tests.db.test_translation_cache_purge` | 3 |
| `tests.app.test_lazy_visualizations` | 3 |
| `tests.api.test_corpus_router` | 3 |
| `tests.api.test_unhandled_exception_route` | 3 |
| `tests.api.test_lti_router` | 3 |
| `tests.api.test_password_change` | 3 |
| `tests.api.test_scan_text_issue_3336` | 3 |
| `tests.app.test_faiss_copy_button_issue_1567` | 3 |
| `tests.app.test_database_download_ui` | 3 |
| `tests.app.test_txt_export_ui` | 3 |
| `tests.core.test_analysis_report_generator` | 3 |
| `tests.core.test_pipeline` | 3 |
| `tests.core.test_semantic_role` | 3 |
| `tests.db.test_auth_lockout` | 3 |
| `tests.core.test_git_scan` | 3 |
| `tests.db.test_auth` | 2 |
| `tests.test_remaining_issues` | 2 |
| `tests.db.test_corpus_db` | 2 |
| `tests.test_validate_hex_color.TestValidateHexColor` | 2 |
| `tests.core.test_ocr_mixed_page_heuristic_issue_2710` | 2 |
| `tests.utils.test_badge_generator` | 2 |
| `tests.db.test_incident_pagination_issue_925` | 2 |
| `tests.api.test_asgi_app` | 2 |
| `tests.db.test_database_backup_restore_security` | 2 |
| `tests.infrastructure.test_fixtures` | 2 |
| `tests.utils.test_filename_length_invariant` | 2 |
| `tests.core.test_fault_tolerance` | 2 |
| `tests.utils.test_pkce_google_oauth_issue_3453` | 2 |
| `tests.core.test_release_large_batch_memory_issue_3479` | 2 |
| `tests.core.test_similarity_engines` | 2 |
| `tests.security.test_document_fingerprint` | 2 |
| `tests.core.test_federated_minhash` | 2 |
| `tests.security.test_metadata_forensics` | 2 |
| `tests.test_normalize_sha256.TestInvalidHashCharacters` | 2 |
| `tests.utils.test_badge_generator_colors` | 2 |
| `tests.app.test_faiss_metric_badge` | 2 |
| `tests.core.test_docx_headings` | 2 |
| `tests.core.test_report_generator` | 2 |
| `tests.api.test_2fa_setup_qr_code_issue_4040` | 2 |
| `tests.api.test_async_scan_progress` | 2 |
| `tests.api.test_scan_stage_metrics` | 2 |
| `tests.app.test_config_backup_ui` | 2 |
| `tests.app.test_faiss_copy_source_issue_1567` | 2 |
| `tests.core.test_api_graph` | 2 |
| `tests.core.test_pptx_scan` | 2 |
| `tests.db.test_failed_password_audit_source_issue_937` | 2 |
| `tests.test_neural_alignment_engine` | 2 |
| `tests.api.test_async_scan_fast_path` | 2 |
| `tests.api.test_login_rate_limiting_issue_2942` | 2 |
| `tests.api.test_validation_errors` | 2 |
| `tests.app.test_app_clear` | 2 |
| `tests.app.test_app_title_config` | 2 |
| `tests.app.test_upload_extension_validation` | 2 |
| `tests.core.test_mock_faiss_index_fixture_issue_3248` | 2 |
| `tests.core.test_revision_burst` | 2 |
| `tests.test_patchwriting_detector` | 2 |
| `tests.utils.test_file_streaming` | 1 |
| `tests.core.test_ai_detector` | 1 |
| `tests.utils.test_warning_list` | 1 |
| `tests.core.test_detect_chunk_language` | 1 |
| `tests.core.test_translator_logger_issue_4093` | 1 |
| `tests.core.test_config` | 1 |
| `tests.core.test_plagiarism_evidence` | 1 |
| `tests.security.test_jwt_utils` | 1 |
| `tests.test_email_regex.TestInvalidEmails` | 1 |
| `tests.core.test_events` | 1 |
| `tests.core.test_load_custom_stopwords_missing_file_issue_2700` | 1 |
| `tests.db.test_security_audit_log` | 1 |
| `tests.utils.test_excel_export` | 1 |
| `tests.core.test_scheduler` | 1 |
| `tests.db.test_password_changed_at_issue_1266` | 1 |
| `tests.db.test_sqlite_timeout_concurrency_documentation_issue_2721` | 1 |
| `tests.app.components` | 1 |
| `tests.core.test_concurrency` | 1 |
| `tests.core.test_rtf_parsing` | 1 |
| `tests.core.test_translator_timeout_issue_3990` | 1 |
| `tests.test_sanitize_filename.TestDangerousCharacters` | 1 |
| `tests.utils.test_report_exporter` | 1 |
| `tests.core.test_metrics_json` | 1 |
| `tests.infrastructure.test_module_docstring_placement` | 1 |
| `tests.utils.test_sso_state_payload_issue_3851` | 1 |
| `tests.utils.test_filename_property` | 1 |
| `tests.core.test_stylometry` | 1 |
| `tests.core.test_token_chunking_issue_3998` | 1 |
| `tests.infrastructure.test_conftest_factories` | 1 |
| `tests.utils.test_posix_path_cache_keys_issue_3028` | 1 |
| `tests.utils.test_sso_csrf_state_validation_issue_2581` | 1 |
| `tests.core.test_ai_scoring_engine` | 1 |
| `tests.core.test_ast_engine` | 1 |
| `tests.core.test_cross_lingual_pipeline` | 1 |
| `tests.core.test_essay_scorer` | 1 |
| `tests.core.test_plagiarism_trends` | 1 |
| `tests.core.test_text_chunking_limits` | 1 |
| `tests.db.test_streaming_snapshot_generator_issue_3405` | 1 |
| `tests.security.test_asgi_app` | 1 |
| `tests.test_bulk_export.TestBulkExportBasicEmpty` | 1 |
| `tests.test_normalize_sha256.TestValidHashes` | 1 |
| `tests.test_ssu_user_profile.TestSSUUserProfileEquality` | 1 |
| `tests.core.test_document_comparison_engine` | 1 |
| `tests.core.test_patchwriting_detector` | 1 |
| `tests.core.test_reliability_engine` | 1 |
| `tests.db.test_password_complexity_zxcvbn` | 1 |
| `tests.infrastructure.test_source_compiles` | 1 |
| `tests.security.test_code_sandbox` | 1 |
| `tests.test_email_regex.TestEdgeCases` | 1 |
| `tests.test_embedding_utils.TestL2NormalizationEdgeCases` | 1 |
| `tests.test_cors` | 1 |
| `tests.test_temp_size_prefix.TestGetTempDirectorySizeBytesEdgeCases` | 1 |
| `tests.test_temp_size_prefix.TestGetTempDirectorySizeBytesPrefix` | 1 |
| `tests.test_unique_word_ratio.TestBasicRatio` | 1 |
| `tests.utils.test_redis_pipeline_clear` | 1 |
| `tests.utils.test_sso_custom_scopes_issue_3458` | 1 |
| `tests.api.test_auth_router` | 1 |
| `tests.app.test_session_warning` | 1 |
| `tests.cli.test_seed_schema_source_issue_681` | 1 |
| `tests.core.test_av_scan` | 1 |
| `tests.core.test_citation_context` | 1 |
| `tests.core.test_code_evasion` | 1 |
| `tests.core.test_cross_modal` | 1 |
| `tests.core.test_discourse_tree` | 1 |
| `tests.core.test_ocr_setting_forwarding` | 1 |
| `tests.core.test_steganography` | 1 |
| `tests.db.test_corpus_db_vacuum` | 1 |
| `tests.db.test_password_changed_at_source_issue_1266` | 1 |
| `tests.db.test_incidents_bulk` | 1 |
| `tests.infrastructure.test_support_security_docs` | 1 |
| `tests.test_chunk_count.TestColumnExistence` | 1 |
| `tests.db.test_soft_delete` | 1 |
| `tests.test_document_index.TestDataIntegrity` | 1 |
| `tests.test_document_versioning` | 1 |
| `tests.test_unique_word_ratio.TestComplexTexts` | 1 |
| `tests.api.test_app_validation` | 1 |
| `tests.app.test_active_sessions_count` | 1 |
| `tests.app.test_faiss_results_ui` | 1 |
| `tests.app.test_storage_widget` | 1 |
| `tests.core.test_code_comment` | 1 |
| `tests.core.test_document_cluster_analyzer` | 1 |
| `tests.core.test_empty_document_error` | 1 |
| `tests.core.test_stopwords_nltk_comparison_issue_2731` | 1 |
| `tests.core.test_text_chunking_issue_2905` | 1 |
| `tests.db.test_incidents_core` | 1 |
| `tests.db.test_incidents_edge_cases` | 1 |
| `tests.db.test_schemas` | 1 |
| `tests.scripts.test_dump_db` | 1 |
| `tests.test_faiss_vector_engine` | 1 |
| `tests.test_multithread_close.TestThreadSafety` | 1 |
| `tests.test_stop_word_utils.TestFilterStopWordsTypes` | 1 |
| `tests.utils.test_sso_integration` | 1 |
| `tests.utils.test_stop_word_utils` | 1 |
| `tests.api.test_rate_limit` | 1 |
| `tests.app.test_app_csv` | 1 |
| `tests.utils.test_sso_avatar_fallback_issue_3459` | 1 |
| `tests.app.test_app_smoke` | 1 |
| `tests.app.test_app_json` | 1 |
| `tests.app.test_app_settings` | 1 |
| `tests.app.test_db_schema_status_ui` | 1 |
| `tests.app.test_doc_mgmt_filter` | 1 |
| `tests.app.test_staged_files_counter` | 1 |
| `tests.core.test_batch_scan` | 1 |
| `tests.core.test_embedding_perf` | 1 |
| `tests.app.test_badge_html_xss` | 1 |
| `tests.core.test_asyncio_policy` | 1 |
| `tests.db.test_filename_security` | 1 |
| `tests.test_otel_tracing_middleware` | 1 |
| `tests.utils.test_temp_manager_exit` | 1 |
