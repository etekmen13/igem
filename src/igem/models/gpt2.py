import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model


class LMScoringWrapper(nn.Module):
    """
    Turns a causal LM into a 4-way classifier by scoring each label word.

    Each item arrives as four (stem, option) continuations; the score of an
    option is the mean log-likelihood of its tokens, and those scores are the
    logits the cross-entropy criterion sees.
    """

    def __init__(self, peft_model, chunk_size=64):
        super().__init__()
        self.model = peft_model
        self.chunk_size = chunk_size

    def forward(self, x):
        input_ids, attention_mask, loss_mask = x[:, :, 0, :], x[:, :, 1, :], x[:, :, 2, :]
        B, O, L = input_ids.shape

        ids = input_ids.reshape(B * O, L).contiguous()
        attn = attention_mask.reshape(B * O, L).contiguous()
        msk = loss_mask.reshape(B * O, L).contiguous()

        scores = torch.zeros(B * O, device=ids.device, dtype=torch.float32)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        for start in range(0, ids.size(0), self.chunk_size):
            end = min(start + self.chunk_size, ids.size(0))

            logits = self.model(
                input_ids=ids[start:end], attention_mask=attn[start:end]
            ).logits

            shift_logits = logits[..., :-1, :].float().contiguous()
            shift_labels = ids[start:end, 1:].contiguous()
            shift_mask = msk[start:end, 1:].float().contiguous()

            token_nll = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
            ).view(shift_labels.shape)

            log_probs = -token_nll * shift_mask
            scores[start:end] = log_probs.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

        return scores.view(B, O)


def get_tokenizer(model_name: str = "gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def get_gpt2_lora(model_name: str = "gpt2-medium", r: int = 8, lora_alpha: int = 32,
                  lora_dropout: float = 0.05) -> nn.Module:
    """GPT-2 with LoRA adapters on c_attn and c_proj, base weights frozen.

    c_proj matches both the attention projection and the MLP projection, which
    is the target set described in Sec. IV-B.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer = get_tokenizer(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model = get_peft_model(
        model,
        LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"],
            # GPT-2 uses Conv1D rather than Linear, so the weights are stored
            # transposed relative to what LoRA assumes
            fan_in_fan_out=True,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    model.print_trainable_parameters()

    return LMScoringWrapper(model).to(device)
