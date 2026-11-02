# legacy

The tools in `htl/` served the earlier study that compared segmentation
inputs by granularity, and the check of the evaluator against the pilot
evaluator. The IPCAI 2027 paper uses none of them. No scorer hashes them.
Their tests stay in `tests/`. Run one from an install of this repository:

```bash
python -m legacy.htl.<module> --help
```

| module | what it does |
|---|---|
| `paired_table`, `arms_paired`, `claims_table`, `leave_one_video_out`, `print_status_tables` | print paired differences between conditions and their marks |
| `kmerge`, `region_counts`, `region_curves` | merge a condition's regions to K, count regions per frame, and lay the scores along that count |
| `condition_inventory`, `check_provenance` | count what ran under each condition, and check each clip's settings, from the records on disk |
| `pilot_check` | compare the evaluator's pilot-mode scores with the pilot evaluator's |
| `d4d_front_behind`, `d4d_front_behind_verdict` | score which of two D4D regions is nearer, and compare an answer with a floor |
