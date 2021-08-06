import logging

from lit_utils.gram_lit_utils import LitGram
from model_utils import get_oecc_model

logger = logging.getLogger(__name__)


class LitOecc(LitGram):
    def get_model(self):
        return get_oecc_model(self.model_name, self.ind_name)
