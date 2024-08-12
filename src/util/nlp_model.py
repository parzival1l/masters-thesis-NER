import sys

# from spacy_download import load_spacy


class NLPModel:
    _model: any
    _provider: str

    def __init__(self, provider: str, model_name: str) -> None:
        self._provider = provider

        if provider == "spacy":
            try:
                self._model = load_spacy(model_name)
            except AttributeError as e:
                sys.exit(e)
        else:
            sys.exit(f"Provider not supported")

    def calculate_similarity(self, query: str, possibilities: list[str]) -> list[float]:
        if self._provider == "spacy":
            _query = self._model(query)
            _sim_results = []
            for possibility in possibilities:
                _possibility = self._model(possibility)
                _sim_results.append((possibility, _query.similarity(_possibility)))
        return _sim_results

    def apply_text_preprocessing(self, input: str, apply_lemma: bool) -> str:
        if self._provider == "spacy":
            input = self._model(input)
            input = " ".join(
                [token.text.lower() for token in input if not token.is_punct and token.text != " "]
            )
            if apply_lemma:
                input = self._model(input)
                input = " ".join([token.lemma_ for token in input])
        return input
