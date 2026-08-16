import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class LocalLLM:
    """
    Local instruction-following LLM used for RAG answer generation.

    The model is intentionally kept separate from retrieval.
    Retrieval provides the evidence; this class only generates
    an answer from the supplied context.
    """

    def __init__(
        self,
        model_name=MODEL_NAME,
        device=None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device
        self.model_name = model_name

        print("=" * 70)
        print("Loading local LLM")
        print("=" * 70)
        print(f"Model : {model_name}")
        print(f"Device: {device}")

        # --------------------------------------------------
        # Tokenizer
        # --------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        if device == "cuda":

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
            )

        else:

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name
            )

        self.model.eval()

        print("LLM loaded successfully.")
        print("=" * 70)

    # ------------------------------------------------------
    # Generation
    # ------------------------------------------------------

    def generate(
        self,
        system_prompt,
        user_prompt,
        max_new_tokens=350,
        temperature=0.0,
    ):
        """
        Generate an answer using the supplied system and user prompts.

        Temperature defaults to 0.0 because this is a factual
        retrieval-augmented QA task rather than creative generation,
        and deterministic output is preferred for research-paper QA.
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        # --------------------------------------------------
        # Apply Qwen chat template
        # --------------------------------------------------

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # --------------------------------------------------
        # Tokenize
        # --------------------------------------------------

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        )

        # Move tensors to model device
        model_device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: value.to(model_device)
            for key, value in inputs.items()
        }

        # --------------------------------------------------
        # Generate
        # --------------------------------------------------

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.9
            generation_kwargs["repetition_penalty"] = 1.05
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        # --------------------------------------------------
        # Remove prompt tokens
        # --------------------------------------------------

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()