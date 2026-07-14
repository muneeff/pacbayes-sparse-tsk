from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class TemporalTheoryAuditError(ValueError):
    """Raised when the temporal PAC-Bayes implementation fails a critical audit."""


@dataclass(frozen=True)
class AuditItem:
    audit_id: str
    route: str
    theoretical_statement: str
    paper_equation: str
    code_location: str
    implemented_quantity: str
    status: str
    severity: str
    qualification: str
    required_paper_wording: str


SERIES_KEY = ["source", "dataset", "series_id"]


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recompute_martingale_certificate(row: pd.Series) -> float:
    empirical = float(row["empirical_gibbs_risk_upper"])
    temperature = float(row["selected_temperature"])
    total_kl = float(row["total_kl_upper"])
    sample_size = int(row["certification_sample_size"])
    delta_series = float(row["delta_series"])
    # The frozen implementation uses a uniform six-point temperature grid.
    temperature_mass = 1.0 / 6.0
    value = (
        empirical
        + temperature / 8.0
        + (
            total_kl
            + np.log(
                1.0
                / (
                    delta_series
                    * temperature_mass
                )
            )
        )
        / (
            temperature
            * sample_size
        )
    )
    return float(min(1.0, value))


def recompute_beta_certificate(row: pd.Series) -> float:
    empirical = float(
        row["empirical_block_gibbs_risk_upper"]
    )
    temperature = float(row["temperature"])
    total_kl = float(row["total_kl_upper"])
    mu = int(row["effective_sample_size"])
    effective_delta = float(row["effective_delta"])
    temperature_mass = 1.0 / 6.0
    value = (
        empirical
        + temperature / 8.0
        + (
            total_kl
            + np.log(
                1.0
                / (
                    effective_delta
                    * temperature_mass
                )
            )
        )
        / (
            temperature
            * mu
        )
    )
    return float(min(1.0, value))


def build_static_audit_items() -> list[AuditItem]:
    return [
        AuditItem(
            "M01",
            "martingale",
            "The prior may depend only on the pre-certification sigma-field G_0.",
            "Prior measurability before Theorem 1",
            "run_temporal_pac_bayes_v2.py: prior subset and prior_diagonal_std",
            "Prior mean, scale, and clipping bound use D_prior only.",
            "PASS",
            "critical",
            "",
            "The martingale prior is G_0-measurable and fixed over the certification horizon.",
        ),
        AuditItem(
            "M02",
            "martingale",
            "The posterior Q_t may be G_t-measurable.",
            "Time-uniform master bound",
            "run_temporal_pac_bayes_v2.py: frozen posterior at final certification horizon",
            "The frozen posterior uses the complete bound+validation trajectory.",
            "PASS_WITH_QUALIFICATION",
            "critical",
            "The code evaluates Q_n only. It does not compute a prefix sequence Q_t.",
            "The theorem is time-uniform, whereas the reported numerical certificate is evaluated only at t=n.",
        ),
        AuditItem(
            "M03",
            "martingale",
            "The empirical loss must lie in [0,1].",
            "Bounded normalized squared loss",
            "pac_bayes_certificate.py: bounded_gibbs_empirical_upper",
            "Targets are clipped by a prior-only bound and squared loss is normalized by 4B^2.",
            "PASS_WITH_QUALIFICATION",
            "critical",
            "The code computes an analytic upper bound on the Gibbs empirical clipped loss, not an exact Monte Carlo expectation.",
            "Report the quantity as an upper bound on clipped normalized Gibbs risk.",
        ),
        AuditItem(
            "M04",
            "martingale",
            "Conditional Hoeffding contributes lambda/8.",
            "Anytime bounded-loss certificate",
            "pac_bayes_certificate.py: anytime_bounded_certificate",
            "certificate = empirical upper + lambda/8 + complexity/(lambda n)",
            "PASS",
            "critical",
            "",
            "The lambda/8 term is the bounded-loss conditional-Hoeffding contribution.",
        ),
        AuditItem(
            "M05",
            "martingale",
            "Temperature selection over a fixed grid requires confidence mass nu(lambda).",
            "Valid temperature grid",
            "pac_bayes_certificate.py: search_certificate",
            "Uniform mass over the six-point frozen temperature grid is included in the logarithmic term.",
            "PASS",
            "critical",
            "",
            "Temperature is selected only from the frozen grid and pays log(1/nu(lambda)).",
        ),
        AuditItem(
            "M06",
            "martingale",
            "The confidence statement must account for both routes and all 210 series.",
            "Familywise implementation allocation",
            "pac_bayes_temporal_dependence_v2.yaml and run_temporal_pac_bayes_v2.py",
            "delta=0.05 is split equally across routes and then divided by 210.",
            "PASS",
            "critical",
            "",
            "The reported development analysis uses familywise route and series allocation.",
        ),
        AuditItem(
            "M07",
            "martingale",
            "Diagonal-Gaussian parameter KL must use the exact Gaussian formula.",
            "Gaussian KL equation",
            "pac_bayes_certificate.py: gaussian_kl_diag",
            "Exact coordinatewise diagonal-Gaussian KL.",
            "PASS",
            "critical",
            "",
            "Parameter complexity is the exact KL between the selected diagonal Gaussian posterior and prior component.",
        ),
        AuditItem(
            "M08",
            "martingale",
            "Selecting among prior components must be charged.",
            "Hierarchical/mixture prior",
            "pac_bayes_certificate.py: search_certificate",
            "KL to one prior component plus log(number of components).",
            "PASS_WITH_QUALIFICATION",
            "major",
            "This is a conservative upper bound on KL to the finite mixture, not the exact mixture KL.",
            "Describe the prior-component term as a valid finite-mixture upper bound.",
        ),
        AuditItem(
            "M09",
            "martingale",
            "Selection of lag, radius, alpha, and rule count must have prior mass.",
            "Deterministic structure KL",
            "pac_bayes_certificate.py: structure_prior_penalty",
            "A finite structure prior is normalized over frozen candidate structures using structural columns only.",
            "PASS_WITH_QUALIFICATION",
            "major",
            "The candidate table is stored in a validation-results file, but the function discards validation losses and uses only frozen structural descriptors.",
            "State that the support and code lengths are G_0-measurable structural quantities; validation losses do not enter the prior mass.",
        ),
        AuditItem(
            "M10",
            "martingale",
            "The theorem directly certifies Gibbs risk; Jensen transfers it to the clipped BMA predictor.",
            "BMA transfer corollary",
            "outputs: certified_object and posterior_mean_parameter_model_certified",
            "The output explicitly says Gibbs/BMA is certified and the posterior-mean parameter model is not.",
            "PASS_WITH_QUALIFICATION",
            "critical",
            "No BMA forecast was evaluated in the predictive benchmark.",
            "Do not describe RMSE of the deterministic posterior-mean parameter model as PAC-Bayes certified.",
        ),
        AuditItem(
            "M11",
            "martingale",
            "The test split must not affect the certificate.",
            "Certification protocol",
            "run_temporal_pac_bayes_v2.py and protocol tests",
            "Only prior, bound, and validation subsets are loaded; test_split_used is false.",
            "PASS",
            "critical",
            "",
            "The test segment is excluded from all certificate calculations.",
        ),
        AuditItem(
            "B01",
            "beta_mixing",
            "For n=2mu a, retain alternating odd blocks and use effective sample size mu.",
            "Odd-block definition",
            "temporal_pac_bayes.py: odd_block_empirical_risk",
            "mu=floor(n/(2a)); the unused tail is discarded and odd blocks are averaged.",
            "PASS",
            "critical",
            "",
            "The beta calculation uses the retained-block effective sample size mu, not n.",
        ),
        AuditItem(
            "B02",
            "beta_mixing",
            "Dependent-to-independent comparison reduces confidence by 2(mu-1) beta(a).",
            "Block beta-mixing bound",
            "temporal_pac_bayes.py: beta_effective_delta",
            "effective_delta = allocated_delta - 2(mu-1) beta_upper.",
            "PASS",
            "critical",
            "",
            "The mixing residual follows the convention used in the paper.",
        ),
        AuditItem(
            "B03",
            "beta_mixing",
            "Post-hoc block-length selection requires confidence allocation over the frozen block grid.",
            "Block-grid implementation corollary",
            "temporal_pac_bayes.py: search_beta_profile",
            "The beta route delta is divided uniformly across block lengths.",
            "PASS",
            "critical",
            "",
            "Each block length pays its frozen grid mass before the mixing residual is deducted.",
        ),
        AuditItem(
            "B04",
            "beta_mixing",
            "Temperature selection also requires confidence mass.",
            "Block/temperature grid corollary",
            "temporal_pac_bayes.py: beta_mixing_certificate_value",
            "The six-point temperature-grid mass is included in the logarithmic term.",
            "PASS",
            "critical",
            "",
            "The beta expression includes both block-grid and temperature-grid confidence costs.",
        ),
        AuditItem(
            "B05",
            "beta_mixing",
            "The prior must be independent of the blocked observations.",
            "Stationary beta-mixing corollary",
            "temporal_pac_bayes.py: zero-mean parameter prior; assumption audit",
            "Consequent-parameter prior is data-independent, but the frozen antecedent feature map was built from an adjacent prior segment.",
            "SENSITIVITY_ONLY",
            "critical",
            "Independence of the complete hypothesis, including the feature map, is not established.",
            "Current beta values are assumption-conditioned sensitivity values, not verified certificates.",
        ),
        AuditItem(
            "B06",
            "beta_mixing",
            "Strict stationarity and a justified finite-sample beta(a) upper envelope are required.",
            "Stationary beta-mixing assumptions",
            "beta_mixing_envelope_profiles.csv and assumption_audit.csv",
            "All geometric envelopes are marked assumption_only; real and synthetic stationarity is unverified.",
            "SENSITIVITY_ONLY",
            "critical",
            "No external verified beta envelope is supplied.",
            "Use the label assumption-conditioned beta-mixing sensitivity throughout.",
        ),
        AuditItem(
            "B07",
            "beta_mixing",
            "Known nonstationary structural-break series are outside the stationary corollary.",
            "Stationarity assumption",
            "pac_bayes_temporal_dependence_v2.yaml",
            "structural_break is marked not_applicable.",
            "PASS",
            "critical",
            "",
            "No beta-mixing certificate is reported for structural-break trajectories.",
        ),
        AuditItem(
            "B08",
            "beta_mixing",
            "Different beta-envelope profiles are different assumptions, not a data-selected confidence grid.",
            "Profile interpretation",
            "run_temporal_pac_bayes_v2.py: best row grouped by series and profile",
            "A best block is selected separately inside each profile; profiles are not minimized jointly.",
            "PASS_WITH_QUALIFICATION",
            "major",
            "A reader must not select the best profile after seeing outcomes and call it one certificate.",
            "Report every envelope profile separately as a sensitivity scenario.",
        ),
        AuditItem(
            "L01",
            "loss_scope",
            "The certified clipped normalized loss is distinct from raw RMSE, MAE, MASE, and sMAPE.",
            "Loss definition and reporting scope",
            "certificate output columns and results tables",
            "The certificate outputs a clipped normalized risk; predictive metrics are stored separately.",
            "PASS_WITH_QUALIFICATION",
            "critical",
            "Target clipping is frequent for Tourism, M4, and structural-break data.",
            "Never translate the certificate numerically into an upper bound on raw RMSE.",
        ),
        AuditItem(
            "C01",
            "confirmatory_scope",
            "A development protocol modified after viewing test outcomes is not confirmatory.",
            "Experimental-status statement",
            "freeze manifests and study_status fields",
            "Development and temporal certificates are labeled exploratory; confirmatory evaluation is reserved.",
            "PASS",
            "critical",
            "",
            "The independent confirmatory run must use unseen series/new seeds and cannot alter the frozen protocol.",
        ),
    ]


def audit_source_contract(
    *,
    temporal_helper_text: str,
    temporal_runner_text: str,
    base_certificate_text: str,
    temporal_config: dict,
    beta_profiles: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []

    required_helper_tokens = [
        "mu = len(loss) // (2 * a)",
        "2.0 * (mu - 1) * float(beta_upper)",
        "allocated_delta - residual",
        "float(temperature) / 8.0",
        "effective_delta",
        "verified_external_upper_bound",
    ]
    for token in required_helper_tokens:
        if token not in temporal_helper_text:
            errors.append(
                f"Missing temporal-helper contract token: {token}"
            )

    required_runner_tokens = [
        '("prior",)',
        '("bound", "validation")',
        '"test_split_used": False',
        '"evaluated_horizon": "final_certification_horizon"',
        '"full_prefix_path_computed": False',
        '"assumption_conditioned_sensitivity"',
    ]
    for token in required_runner_tokens:
        if token not in temporal_runner_text:
            errors.append(
                f"Missing temporal-runner contract token: {token}"
            )

    required_base_tokens = [
        "float(temperature) / 8.0",
        "temperature_mass",
        "gaussian_kl_diag",
        "prior_component_penalty",
    ]
    for token in required_base_tokens:
        if token not in base_certificate_text:
            errors.append(
                f"Missing base-certificate contract token: {token}"
            )

    confidence = temporal_config["confidence"]
    if not np.isclose(
        float(confidence["martingale_route_fraction"])
        + float(confidence["beta_route_fraction"]),
        1.0,
    ):
        errors.append(
            "Martingale and beta route fractions do not sum to one."
        )
    if int(confidence["familywise_series_count"]) != 210:
        errors.append(
            "Familywise series count is not 210."
        )

    if not (
        beta_profiles["envelope_verification_status"]
        .astype(str)
        .eq("assumption_only")
        .all()
    ):
        errors.append(
            "At least one supplied beta profile is not marked assumption_only."
        )
    if not (
        beta_profiles["source_reference"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .all()
    ):
        errors.append(
            "An assumption-only beta profile unexpectedly contains a verification source."
        )
    return errors


def audit_result_contract(
    *,
    martingale: pd.DataFrame,
    beta_sensitivity: pd.DataFrame,
    beta_best: pd.DataFrame,
    assumptions: pd.DataFrame,
    expected_series: int = 210,
    tolerance: float = 1.0e-10,
) -> list[str]:
    errors: list[str] = []

    if len(martingale) != expected_series:
        errors.append(
            f"Expected {expected_series} martingale rows; found {len(martingale)}."
        )
    if "status" not in martingale.columns:
        errors.append("Martingale result has no status column.")
        return errors

    failures = int(
        (~martingale["status"].astype(str).eq("PASS")).sum()
    )
    if failures:
        errors.append(
            f"Martingale result contains {failures} failures."
        )

    passed = martingale.loc[
        martingale["status"].astype(str).eq("PASS")
    ].copy()
    if not passed.empty:
        if passed["test_split_used"].fillna(True).astype(bool).any():
            errors.append("Martingale result reports test usage.")
        values = pd.to_numeric(
            passed["martingale_certificate"],
            errors="coerce",
        ).to_numpy(float)
        if not np.all(np.isfinite(values)):
            errors.append(
                "Martingale certificates contain non-finite values."
            )

        recomputed = passed.apply(
            recompute_martingale_certificate,
            axis=1,
        ).to_numpy(float)
        if not np.allclose(
            values,
            recomputed,
            rtol=0.0,
            atol=tolerance,
        ):
            maximum = float(
                np.max(np.abs(values - recomputed))
            )
            errors.append(
                "Martingale equation/code recomputation mismatch; "
                f"maximum absolute error={maximum}."
            )

    if not beta_best.empty:
        verified = int(
            beta_best[
                "reported_as_verified_certificate"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if verified != 0:
            errors.append(
                f"Expected zero verified beta rows; found {verified}."
            )

    beta_pass = beta_sensitivity.loc[
        beta_sensitivity["status"].astype(str).eq("PASS")
    ].copy()
    if not beta_pass.empty:
        if beta_pass[
            "test_split_used"
        ].fillna(True).astype(bool).any():
            errors.append(
                "Beta sensitivity reports test usage."
            )
        recorded = pd.to_numeric(
            beta_pass["certificate"],
            errors="coerce",
        ).to_numpy(float)
        recomputed = beta_pass.apply(
            recompute_beta_certificate,
            axis=1,
        ).to_numpy(float)
        if not np.allclose(
            recorded,
            recomputed,
            rtol=0.0,
            atol=tolerance,
        ):
            maximum = float(
                np.max(np.abs(recorded - recomputed))
            )
            errors.append(
                "Beta equation/code recomputation mismatch; "
                f"maximum absolute error={maximum}."
            )

    structural = assumptions.loc[
        assumptions["dataset"].astype(str).eq(
            "structural_break"
        )
    ]
    if structural.empty:
        errors.append(
            "Assumption audit has no structural_break row."
        )
    elif not structural[
        "beta_route_applicability"
    ].astype(str).eq("not_applicable").all():
        errors.append(
            "structural_break is not marked beta not_applicable."
        )

    return errors


def audit_items_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [asdict(item) for item in build_static_audit_items()]
    )


def overall_audit_status(
    items: pd.DataFrame,
    critical_errors: Iterable[str],
) -> str:
    errors = list(critical_errors)
    if errors:
        return "FAIL"
    if items["status"].astype(str).isin(
        ["PASS_WITH_QUALIFICATION", "SENSITIVITY_ONLY"]
    ).any():
        return "PASS_WITH_QUALIFICATIONS"
    return "PASS"


def write_markdown_report(
    items: pd.DataFrame,
    *,
    overall_status: str,
    critical_errors: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Temporal PAC-Bayes V2: Equation-to-Code Audit",
        "",
        f"**Overall status:** `{overall_status}`",
        "",
        "## Audit scope",
        "",
        "The audit checks the frozen final-horizon martingale certificate, "
        "the assumption-conditioned beta-mixing block analysis, confidence "
        "allocation, test exclusion, loss scope, and confirmatory status.",
        "",
    ]

    if critical_errors:
        lines.extend(
            [
                "## Critical errors",
                "",
                *[f"- {error}" for error in critical_errors],
                "",
            ]
        )

    lines.extend(
        [
            "## Equation-level correspondence",
            "",
            "| ID | Route | Status | Theory | Code | Qualification |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in items.itertuples(index=False):
        qualification = (
            str(row.qualification).replace("|", "\\|")
            if str(row.qualification)
            else ""
        )
        theory_text = str(
            row.theoretical_statement
        ).replace("|", "\\|")
        code_text = str(
            row.code_location
        ).replace("|", "\\|")
        lines.append(
            f"| {row.audit_id} | {row.route} | {row.status} | "
            f"{theory_text} | "
            f"{code_text} | "
            f"{qualification} |"
        )

    lines.extend(
        [
            "",
            "## Binding interpretation",
            "",
            "1. The martingale/time-uniform route is the primary theoretical route.",
            "2. Its numerical value is evaluated only at the final certification horizon.",
            "3. The empirical term is an analytic upper bound on clipped normalized Gibbs loss.",
            "4. The certificate does not upper-bound raw RMSE, MAE, MASE, or sMAPE.",
            "5. The beta-mixing outputs remain sensitivity values because stationarity, "
            "feature-map independence, and verified beta envelopes are not established.",
            "6. The development models remain test-informed; a later unseen-data run is "
            "required for confirmatory claims.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
