
import torch

from transformers import (
    AutoModelForQuestionAnswering
)

from config.config import (
    MODEL_NAME,
    DEVICE
)


class LegalClauseQAModel:

    def __init__(
        self,
        model_name=MODEL_NAME,
        device=DEVICE
    ):

        self.model_name = model_name
        self.device = device

        print(f"Loading QA model: {model_name}")

        self.model = (
            AutoModelForQuestionAnswering.from_pretrained(
                model_name
            )
        )

        self.model.to(self.device)

        print(f"Model loaded on: {self.device}")

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        self.model.train()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return outputs

    def predict(
        self,
        input_ids,
        attention_mask
    ):

        self.model.eval()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        return outputs

    def save(self, save_directory):

        self.model.save_pretrained(
            save_directory
        )

        print(
            f"Model saved to: {save_directory}"
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_directory,
        device=DEVICE
    ):

        instance = cls.__new__(cls)

        instance.model_name = str(
            checkpoint_directory
        )

        instance.device = device

        instance.model = (
            AutoModelForQuestionAnswering.from_pretrained(
                checkpoint_directory
            )
        )

        instance.model.to(device)

        print(
            f"Checkpoint loaded from: {checkpoint_directory}"
        )

        return instance


if __name__ == "__main__":

    qa_model = LegalClauseQAModel()

    print("=" * 55)
    print("QA MODEL INFORMATION")
    print("=" * 55)

    print(
        f"Architecture: {qa_model.model.config.model_type}"
    )

    print(
        f"Hidden size: {qa_model.model.config.hidden_size}"
    )

    print(
        f"Layers: {qa_model.model.config.num_hidden_layers}"
    )

    print(
        f"Parameters: {sum(p.numel() for p in qa_model.model.parameters()):,}"
    )

    print(
        f"Device: {qa_model.device}"
    )

    print("Model initialization successful.")
