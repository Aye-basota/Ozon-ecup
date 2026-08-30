# Reconstruction validation

Overall: **PASS**

| Check | Result | Detail |
|---|---|---|
| required_paths_present | PASS | all present |
| experiment_csv_jsonl_row_parity | PASS | {"csv":138,"jsonl":138} |
| global_experiment_ids_unique | PASS | [] |
| required_registry_fields_nonblank | PASS | {"experiment_id":0,"canonical_name":0,"family":0,"date":0,"parent_baseline":0,"change":0,"model_family":0,"validation_protocol":0,"cv_score":0,"delta_cv":0,"folds_positive":0,"folds_total":0,"lb_score":0,"runtime":0,"status":0,"evidence_strength":0,"artifacts":0,"duplicate_of":0,"compatible_tags":0,"notes":0} |
| unknown_is_explicit | PASS | no blank CSV cells |
| report_catalog_registry_parity | PASS | {"catalog":124,"registry":124} |
| one_design_evidence_record_per_primary_report | PASS | {"design_rows":124,"unique_ids":124,"matched_report_ids":124,"schema_valid":true} |
| one_normalized_card_per_registry_row | PASS | {"cards":138,"registry":138} |
| one_family_page_per_family | PASS | {"pages":12,"families":12} |
| confirmed_lb_artifacts_in_submission_registry | PASS | 11 links matched |
| confirmed_lb_has_no_platform_overclaim | PASS | all marked repository-internal only |
| all_existing_submissions_have_forensic_recipe | PASS | 36 recipes present |
| excluded_interpretive_sources_not_used | PASS | zero rows |
| secondary_summaries_are_conflict_objects_not_fact_sources | PASS | {"secondary_audit_rows":22,"contradiction_registry_rows":22,"all_marked_used_for_facts_no":true} |
| derived_json_and_jsonl_parse | PASS | all parsed |
| source_and_linked_worktrees_untouched | PASS | {"source_untouched_verified":true,"main_files":3599,"linked_worktrees":6} |
| no_unnecessary_large_artifacts_copied | PASS | no file above 50 MiB |
