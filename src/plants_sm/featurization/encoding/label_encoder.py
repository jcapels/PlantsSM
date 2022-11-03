from plants_sm.featurization.encoding.encoder import Encoder


class LabelEncoder(Encoder):

    def _set_tokens(self, i: int, token: str):
        self.tokens[token] = i + 1
