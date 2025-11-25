from transformers import AutoConfig, AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL

DEFAULT_MODEL_NAME = "prajjwal1/bert-tiny"

def create_model(model_name: str = DEFAULT_MODEL_NAME):
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    config.output_attentions = False
    config.output_hidden_states = False

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )
    return model
