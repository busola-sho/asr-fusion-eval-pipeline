class ASRFusionPipeline():
    def __init__(self, models, fusion_strategy, sentence_confidence=None, evaluator=None):
        self.asr_models=models
        self.fusion_strategy=fusion_strategy
        self.sentence_confidence=None
        self.evaluator=None
    
    def load_models(self):
        for model in self.asr_models:
            model.load()

    def run(self, audio, sample_rate):
        hypotheses=[model.transcribe(audio) for model in self.asr_models]
        return self.fusion_strategy.fuse(hypotheses)



    
