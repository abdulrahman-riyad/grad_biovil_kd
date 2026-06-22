from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalStudentRun:
    key: str
    display_name: str
    image_student: str
    text_encoder: str
    final_run_name: str
    stage2_run_name: str
    stage1_checkpoint_candidates: tuple[str, ...]


def stage1_checkpoint_candidates(run_key: str) -> tuple[str, ...]:
    return (
        (
            "checkpoints/contrastive_baselines/full_gallery_stage1/"
            f"{run_key}_stage1_contrastive/best.pt"
        ),
    )


FINAL_STUDENT_RUNS: dict[str, FinalStudentRun] = {
    "mobilevit_clinical_distilbert": FinalStudentRun(
        key="mobilevit_clinical_distilbert",
        display_name="MobileViT + ClinicalDistilBERT",
        image_student="mobilevit",
        text_encoder="clinical_distilbert",
        final_run_name="mobilevit_clinical_distilbert_full_student_kd_hn",
        stage2_run_name="mobilevit_clinical_distilbert_stage2_hard_negative",
        stage1_checkpoint_candidates=stage1_checkpoint_candidates("mobilevit_clinical_distilbert"),
    ),
    "repvit_clinical_distilbert": FinalStudentRun(
        key="repvit_clinical_distilbert",
        display_name="RepViT + ClinicalDistilBERT",
        image_student="repvit",
        text_encoder="clinical_distilbert",
        final_run_name="repvit_clinical_distilbert_full_student_kd_hn",
        stage2_run_name="repvit_clinical_distilbert_stage2_hard_negative",
        stage1_checkpoint_candidates=stage1_checkpoint_candidates("repvit_clinical_distilbert"),
    ),
    "mobilevit_distil_biobert": FinalStudentRun(
        key="mobilevit_distil_biobert",
        display_name="MobileViT + DistilBioBERT",
        image_student="mobilevit",
        text_encoder="distil_biobert",
        final_run_name="mobilevit_distil_biobert_full_student_kd_hn",
        stage2_run_name="mobilevit_distil_biobert_stage2_hard_negative",
        stage1_checkpoint_candidates=stage1_checkpoint_candidates("mobilevit_distil_biobert"),
    ),
    "repvit_distil_biobert": FinalStudentRun(
        key="repvit_distil_biobert",
        display_name="RepViT + DistilBioBERT",
        image_student="repvit",
        text_encoder="distil_biobert",
        final_run_name="repvit_distil_biobert_full_student_kd_hn",
        stage2_run_name="repvit_distil_biobert_stage2_hard_negative",
        stage1_checkpoint_candidates=stage1_checkpoint_candidates("repvit_distil_biobert"),
    ),
}


FINAL_STUDENT_RUN_ORDER: list[str] = [
    "mobilevit_clinical_distilbert",
    "repvit_clinical_distilbert",
    "mobilevit_distil_biobert",
    "repvit_distil_biobert",
]
