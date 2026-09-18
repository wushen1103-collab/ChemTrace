import json
import hashlib
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def test_released_certificate_matches_schema() -> None:
    schema = json.loads((ROOT / "schemas/chemtrace-certificate.schema.json").read_text(encoding="utf-8"))
    certificate = json.loads(
        (
            ROOT
            / "reports/certificates/phase0_tf_bbbp_stereo_changed_aff82_d20.final-manuscript.certificate.json"
        ).read_text(encoding="utf-8")
    )
    assert certificate["schema_sha256"] == hashlib.sha256(
        (ROOT / "schemas/chemtrace-certificate.schema.json").read_bytes()
    ).hexdigest()
    Draft202012Validator(schema).validate(certificate)
