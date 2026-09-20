from asr_pipeline.schemas import EvaluationResult
from jiwer import wer
from asr_pipeline.evaluation.meaning import score_meaning_alteration


class Evaluator:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def evaluate(
        self,
        prediction: str,
        reference: str,
    ) -> EvaluationResult:

        wer_score = wer_score = wer(reference, prediction)

        meaning_score = score_meaning_alteration(
            client=self.client,
            model_name=self.model_name,
            reference=reference,
            hypothesis=prediction,
        )

        return EvaluationResult(
            wer=wer_score,
            meaning_score=meaning_score,
        )