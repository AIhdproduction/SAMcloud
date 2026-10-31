import json

from samcloud.engine.run_summary import RunSummary


def test_run_summary_includes_metrics_and_stage_durations(tmp_path):
    products = tmp_path / "demo" / "products"
    dense = products / "dense"
    classified = products / "classified" / "classified.ply"
    dense.mkdir(parents=True)
    classified.parent.mkdir(parents=True)
    (dense / "fused.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 12\nend_header\n")
    classified.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 12\nend_header\n")
    reports = products.parent / "reports"
    reports.mkdir()
    summary_file = reports / "classification_summary.json"
    summary_file.write_text(json.dumps({
        "point_count": 12,
        "unknown_points": 3,
        "images_processed": 5,
        "class_counts": {"ground": 9, "Unknown/noise": 3},
    }), encoding="utf-8")

    summary = RunSummary(products, {
        "image_count": 5,
        "capture_type": "drone_gps",
        "gps_mode": "RTK",
        "gps_max_error_m": 0.1,
        "gpu": True,
        "dense_max_image_size": 3200,
        "dense_num_iterations": 3,
        "class_set": "outdoor",
        "min_votes": 1,
        "sam_tile_size": 1008,
        "sam_tile_min_overlap": 0.2,
        "classified_output": str(classified),
        "classification_summary": str(summary_file),
    })
    summary.record_stage("Dense stereo", 12.2)

    report = summary.write()

    content = report.read_text(encoding="utf-8")
    assert "Dense stereo: 00:00:12" in content
    assert "Dense points: 12" in content
    assert "Classified coverage: 75.00%" in content
    assert "ground: 9 (75.00%)" in content
