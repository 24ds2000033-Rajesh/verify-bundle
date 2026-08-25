import hashlib
import json
import math
import re
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler


REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]

UNSAFE_EXTENSIONS = {
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
}

BASE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

CARD_PREFIX = "<!-- tds-model-card "
CARD_SUFFIX = " -->"


def utf8_bytes(value):
    return value.encode("utf-8")


def sha256_hex(value):
    if isinstance(value, str):
        value = utf8_bytes(value)
    return hashlib.sha256(value).hexdigest()


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def is_nonempty_string(value):
    return isinstance(value, str) and len(value) > 0


def is_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and value <= 9007199254740991
    )


def is_finite_unit_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def add_violation(violations, code):
    violations.add(code)


def parse_json_file(files, name):
    """
    Parse a UTF-8 string as JSON.

    Returns:
      (parsed_value, None) on success
      (None, violation_code) on failure
    """
    if name not in files:
        return None, None

    value = files[name]

    if not isinstance(value, str):
        return None, f"INVALID_FILE:{name}"

    try:
        return json.loads(
            value,
            object_pairs_hook=OrderedDict,
        ), None
    except Exception:
        return None, f"INVALID_JSON:{name}"


def validate_policy(policy):
    if not isinstance(policy, dict):
        return False

    required = policy.get("requiredSlices")

    if not isinstance(required, list) or len(required) == 0:
        return False

    if any(not is_nonempty_string(x) for x in required):
        return False

    if len(set(required)) != len(required):
        return False

    for field in ("license", "intendedUse", "limitations"):
        if not is_nonempty_string(policy.get(field)):
            return False

    return True


def validate_inventory(files, inventory_value, violations):
    """
    inventory.json must be a compact JSON array.

    Each entry must have exactly these keys, in this order:
        name, bytes, sha256

    It must describe every file except inventory.json and no others.
    """

    if not isinstance(inventory_value, list):
        add_violation(violations, "INVENTORY_MISMATCH")
        return

    expected_names = sorted(
        [name for name in files.keys() if name != "inventory.json"],
        key=lambda x: utf8_bytes(x),
    )

    actual_names = []

    for entry in inventory_value:
        if not isinstance(entry, OrderedDict):
            add_violation(violations, "INVENTORY_MISMATCH")
            continue

        if list(entry.keys()) != ["name", "bytes", "sha256"]:
            add_violation(violations, "INVENTORY_MISMATCH")
            continue

        name = entry["name"]

        if not isinstance(name, str):
            add_violation(violations, "INVENTORY_MISMATCH")
            continue

        actual_names.append(name)

        if name not in files or name == "inventory.json":
            add_violation(violations, "INVENTORY_MISMATCH")
            continue

        actual_bytes = len(utf8_bytes(files[name]))
        actual_sha = sha256_hex(files[name])

        if entry["bytes"] != actual_bytes:
            add_violation(violations, "INVENTORY_MISMATCH")

        if (
            not isinstance(entry["sha256"], str)
            or entry["sha256"] != actual_sha
            or entry["sha256"].lower() != entry["sha256"]
        ):
            add_violation(violations, "INVENTORY_MISMATCH")

    if actual_names != expected_names:
        add_violation(violations, "INVENTORY_MISMATCH")

    # Check duplicate names independently.
    if len(actual_names) != len(set(actual_names)):
        add_violation(violations, "INVENTORY_MISMATCH")

    # Inventory itself must be compact canonical JSON.
    try:
        canonical = compact_json(inventory_value)
        if files["inventory.json"] != canonical:
            add_violation(violations, "INVENTORY_MISMATCH")
    except Exception:
        add_violation(violations, "INVENTORY_MISMATCH")


def inventory_digest(files):
    """
    Build the inventory from recomputed exact UTF-8 bytes, excluding
    inventory.json itself, sorted by UTF-8 filename.
    """

    names = sorted(
        [name for name in files.keys() if name != "inventory.json"],
        key=lambda x: utf8_bytes(x),
    )

    rebuilt = []

    for name in names:
        raw = utf8_bytes(files[name])
        rebuilt.append(
            OrderedDict(
                [
                    ("name", name),
                    ("bytes", len(raw)),
                    ("sha256", hashlib.sha256(raw).hexdigest()),
                ]
            )
        )

    return sha256_hex(compact_json(rebuilt))


def validate_adapter_config(config, violations):
    if not isinstance(config, dict):
        add_violation(violations, "INVALID_ADAPTER_CONFIG")
        return

    r = config.get("r")
    targets = config.get("target_modules")

    if not is_safe_integer(r):
        add_violation(violations, "INVALID_ADAPTER_CONFIG")

    if (
        not isinstance(targets, list)
        or len(targets) == 0
        or any(not is_nonempty_string(x) for x in targets)
        or len(set(targets)) != len(targets)
    ):
        add_violation(violations, "INVALID_ADAPTER_CONFIG")


def validate_training_manifest(manifest, violations):
    if not isinstance(manifest, dict):
        add_violation(violations, "INVALID_TRAINING_MANIFEST")
        return

    base_revision = manifest.get("baseRevision")

    if not isinstance(base_revision, str):
        add_violation(violations, "MUTABLE_BASE_REVISION")
    elif not BASE_REVISION_RE.fullmatch(base_revision):
        add_violation(violations, "MUTABLE_BASE_REVISION")

    required_fields = [
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    ]

    for field in required_fields:
        if not is_nonempty_string(manifest.get(field)):
            add_violation(
                violations,
                f"MISSING_MANIFEST_FIELD:{field}",
            )


def validate_evaluation(
    evaluation,
    required_slices,
    expected_model_digest,
    violations,
):
    if not isinstance(evaluation, dict):
        add_violation(violations, "INVALID_EVALUATION")
        return

    # Evaluation must bind the model artifact.
    evaluation_model_digest = evaluation.get("modelArtifactDigest")

    if evaluation_model_digest != expected_model_digest:
        add_violation(violations, "MODEL_ARTIFACT_MISMATCH")

    if "aggregate" not in evaluation:
        add_violation(violations, "INVALID_AGGREGATE")
    elif not is_finite_unit_number(evaluation["aggregate"]):
        add_violation(violations, "INVALID_AGGREGATE")

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):
        for name in required_slices:
            add_violation(violations, f"MISSING_SLICE:{name}")
        return

    for name in required_slices:
        if name not in slices:
            add_violation(violations, f"MISSING_SLICE:{name}")
            continue

        if not is_finite_unit_number(slices[name]):
            add_violation(violations, f"SLICE_RANGE:{name}")


def validate_model_card(
    readme,
    policy,
    manifest,
    violations,
):
    if not isinstance(readme, str):
        add_violation(violations, "INVALID_FILE:README.md")
        return

    count = readme.count(CARD_PREFIX)

    if count == 0:
        add_violation(violations, "MODEL_CARD_COUNT")
        add_violation(violations, "MISSING_MODEL_CARD")
        return

    if count > 1:
        add_violation(violations, "MODEL_CARD_COUNT")
        return

    start = readme.find(CARD_PREFIX)

    payload_start = start + len(CARD_PREFIX)
    end = readme.find(CARD_SUFFIX, payload_start)

    if end == -1:
        add_violation(violations, "INVALID_MODEL_CARD")
        return

    payload = readme[payload_start:end]

    try:
        card = json.loads(
            payload,
            object_pairs_hook=OrderedDict,
        )
    except Exception:
        add_violation(violations, "INVALID_MODEL_CARD")
        return

    if not isinstance(card, dict):
        add_violation(violations, "INVALID_MODEL_CARD")
        return

    expected = {
        "task": manifest.get("task"),
        "baseRevision": manifest.get("baseRevision"),
        "datasetDigest": manifest.get("datasetDigest"),
        "modelArtifactDigest": manifest.get("modelArtifactDigest"),
        "license": policy.get("license"),
        "intendedUse": policy.get("intendedUse"),
        "limitations": policy.get("limitations"),
    }

    for field, expected_value in expected.items():
        if card.get(field) != expected_value:
            add_violation(violations, "MODEL_CARD_MISMATCH")
            break


def verify_bundle(body):
    violations = set()

    policy = body.get("policy")
    files = body.get("files")

    if not validate_policy(policy):
        add_violation(violations, "INVALID_POLICY")

    # Required input-level validation is handled by the caller for files.
    if not isinstance(files, dict):
        add_violation(violations, "INVALID_POLICY")
        return violations, None

    # Files must all have string names and UTF-8-string values.
    for name, value in files.items():
        if not isinstance(name, str) or not isinstance(value, str):
            if isinstance(name, str):
                add_violation(violations, f"INVALID_FILE:{name}")
            continue

    # Missing files.
    for name in REQUIRED_FILES:
        if name not in files:
            add_violation(violations, f"MISSING_FILE:{name}")

    # Extra files.
    for name in files:
        if name not in REQUIRED_FILES:
            add_violation(violations, "UNTRACKED_FILE")

    # Unsafe weight extensions.
    for name in files:
        if isinstance(name, str):
            lower = name.lower()
            for extension in UNSAFE_EXTENSIONS:
                if lower.endswith(extension):
                    add_violation(violations, "UNSAFE_WEIGHTS")
                    break

    # Parse machine-readable files.
    inventory = None
    config = None
    manifest = None
    evaluation = None

    if "inventory.json" in files:
        inventory, error = parse_json_file(files, "inventory.json")
        if error:
            add_violation(violations, error)

    if "adapter_config.json" in files:
        config, error = parse_json_file(files, "adapter_config.json")
        if error:
            add_violation(violations, error)

    if "training_manifest.json" in files:
        manifest, error = parse_json_file(
            files,
            "training_manifest.json",
        )
        if error:
            add_violation(violations, error)

    if "evaluation.json" in files:
        evaluation, error = parse_json_file(
            files,
            "evaluation.json",
        )
        if error:
            add_violation(violations, error)

    # Inventory verification.
    if (
        "inventory.json" in files
        and "inventory.json" not in {
            code.split(":", 1)[1]
            for code in violations
            if code.startswith("INVALID_JSON:")
            and ":" in code
        }
    ):
        validate_inventory(files, inventory, violations)

    # Adapter config.
    if (
        "adapter_config.json" in files
        and not any(
            code == "INVALID_JSON:adapter_config.json"
            for code in violations
        )
    ):
        validate_adapter_config(config, violations)

    # Training manifest.
    if (
        "training_manifest.json" in files
        and not any(
            code == "INVALID_JSON:training_manifest.json"
            for code in violations
        )
    ):
        validate_training_manifest(manifest, violations)

    # Artifact digests.
    model_digest = None
    evaluation_digest = None

    if "adapter_model.safetensors" in files:
        model_digest = sha256_hex(
            utf8_bytes(files["adapter_model.safetensors"])
        )

    if "evaluation.json" in files:
        evaluation_digest = sha256_hex(
            utf8_bytes(files["evaluation.json"])
        )

    if isinstance(manifest, dict):
        if model_digest is not None:
            if manifest.get("modelArtifactDigest") != model_digest:
                add_violation(
                    violations,
                    "MODEL_ARTIFACT_MISMATCH",
                )

        if evaluation_digest is not None:
            if manifest.get("evaluationArtifactDigest") != evaluation_digest:
                add_violation(
                    violations,
                    "EVALUATION_ARTIFACT_MISMATCH",
                )

    # Evaluation.
    if (
        evaluation is not None
        and isinstance(manifest, dict)
        and isinstance(policy, dict)
        and model_digest is not None
    ):
        validate_evaluation(
            evaluation,
            policy.get("requiredSlices", []),
            model_digest,
            violations,
        )

    # Model card.
    if "README.md" in files:
        if isinstance(manifest, dict) and isinstance(policy, dict):
            validate_model_card(
                files["README.md"],
                policy,
                manifest,
                violations,
            )
        else:
            # If machine metadata cannot be established, the card cannot
            # be validated consistently.
            validate_model_card(
                files["README.md"],
                policy if isinstance(policy, dict) else {},
                manifest if isinstance(manifest, dict) else {},
                violations,
            )

    digest = inventory_digest(files)

    return violations, digest


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, obj):
        raw = json.dumps(
            obj,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path != "/verify-bundle":
            self._send_json(
                404,
                {"error": "NOT_FOUND"},
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))

            if length <= 0:
                raise ValueError()

            raw = self.rfile.read(length)

            body = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=OrderedDict,
            )

        except Exception:
            self._send_json(
                400,
                {"error": "INVALID_INPUT"},
            )
            return

        # The request itself must be an object.
        if not isinstance(body, dict):
            self._send_json(
                400,
                {"error": "INVALID_INPUT"},
            )
            return

        # Missing policy or non-object files are specifically HTTP 400.
        if "policy" not in body:
            self._send_json(
                400,
                {"error": "INVALID_INPUT"},
            )
            return

        if "files" not in body or not isinstance(body["files"], dict):
            self._send_json(
                400,
                {"error": "INVALID_INPUT"},
            )
            return

        # Policy itself must be an object at request level.
        if not isinstance(body["policy"], dict):
            self._send_json(
                400,
                {"error": "INVALID_INPUT"},
            )
            return

        try:
            violations, digest = verify_bundle(body)

            # UTF-8 byte ordering.
            sorted_violations = sorted(
                set(violations),
                key=lambda x: utf8_bytes(x),
            )

            response = OrderedDict(
                [
                    (
                        "decision",
                        "admit"
                        if len(sorted_violations) == 0
                        else "reject",
                    ),
                    ("violations", sorted_violations),
                    ("inventoryDigest", digest),
                ]
            )

            self._send_json(200, response)

        except Exception:
            # The verifier should not leak implementation details.
            self._send_json(
                200,
                {
                    "decision": "reject",
                    "violations": ["INVALID_POLICY"],
                    "inventoryDigest": None,
                },
            )


# Vercel's Python runtime looks for the module-level handler.
