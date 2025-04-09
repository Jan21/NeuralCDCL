import torch
from litgpt import LLM
from cdcl.env import AutoregressiveCDCLEnvironment


class InferenceRunner:
    def __init__(self, model: LLM, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature

    def run(self, env: AutoregressiveCDCLEnvironment, max_steps: int) -> list[list[int]]:
        for _ in range(max_steps):
            current_input = env.get_current_input()
            input_tensor = torch.tensor(current_input, device=self.model.device).unsqueeze(0)

            logits = self.model(input_tensor)[0][:, -1, :]  # shape: [1, vocab_size]

            # Apply temperature + sample next token
            probs = torch.nn.functional.softmax(logits / self.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Let environment process it
            env.append(next_token)

            if env.is_finished():
                break

        return env.get_current_input()
