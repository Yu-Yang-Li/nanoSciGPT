import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "data" / "precomputed_results"
DOMAINS = {
    "text",
    "dna",
    "protein",
    "smiles",
    "weather",
    "crystal",
    "structure3d",
    "image",
    "spectrum",
    "field",
}


def test_all_classroom_domains_have_portable_precomputed_results() -> None:
    manifest = json.loads((RESULT_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert set(manifest["domains"]) == DOMAINS
    for domain, entry in manifest["domains"].items():
        result_path = ROOT / entry["result_file"]
        assert result_path.is_file(), f"missing fallback result for {domain}"

        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["schema_version"] == 1
        assert result["artifact_type"] == "precomputed_classroom_fallback"
        assert result["domain"] == domain
        assert result["status"] == "completed"
        assert result["profile"] == "smoke"
        assert result["teaching_only"] is True
        assert result["pretraining"]["name"]
        assert result["downstream"]["metric_name"]
        assert isinstance(result["downstream"]["metric_value"], (int, float))

        serialized = json.dumps(result, ensure_ascii=False)
        assert ":\\\\" not in serialized
        assert '"/' not in serialized
