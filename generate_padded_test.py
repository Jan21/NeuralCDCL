import torch

# Improved Mock Model to simulate transformer behavior
class MockModel:
    def __init__(self, control_tokens):
        self.control_tokens = control_tokens
        self.device = "cpu"  # Ensure CPU compatibility
        self.tokenizer = self.MockTokenizer()  # Fake tokenizer
    
    class MockTokenizer:
        bos_token_id = 0  # Define fake [BOS] token

    def eval(self):
        """Simulates evaluation mode"""
        pass

    def __call__(self, input_ids):
        """Fake transformer output (forces transitions to UP_START or AC_START)"""
        batch_size, _ = input_ids.shape
        logits = torch.zeros(batch_size, 11)  # Fake logits
        logits[:, self.control_tokens["up_tokens"]["arguments"]] = 1.0  # Prioritize UP_ARGUMENTS
        logits[:, self.control_tokens["up_tokens"]["results"]] = 2.0  # Prioritize UP_ARGUMENTS
        return logits.unsqueeze(1)  # Simulate model shape

    def generate_packed(
        self,
        inputs: list[torch.Tensor],
        max_length: int,
        stop_token: int,
        temperature: float = 1.0,
    ) -> list[torch.Tensor]:
        """
        Generate text while dynamically adjusting context when encountering UNIT_PROPAGATION_START or ANALYZE_CONFLICT_START.
        """
        self.eval()
        generated_sequences = []

        for input_ids in inputs:
            current_input = torch.tensor(input_ids, device=self.device).unsqueeze(0)  # (1, seq_len)
            generated = current_input
            saved_context = None  # To store original context before modifying

            print('\n\nNEW')
            print(input_ids)

            while True:
                for _ in range(max_length - generated.size(1)):
                    with torch.no_grad():
                        logits = self(generated)[:, -1, :]  # (1, vocab_size)

                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                    generated = torch.cat([generated, next_token], dim=-1)

                    print(f'Next token = {next_token}')
                    print(f'UP SART  = {self.control_tokens["up_tokens"]["start"]}')

                    if next_token.item() == stop_token:
                        print('break 1')
                        break

                    # Detect transition to UP or AC and store context
                    if next_token.item() in [self.control_tokens["up_tokens"]["start"]]:
                        saved_context = generated.clone()  # Save context so far
                        print(f'break 2, saving {saved_context}')
                        break  # Stop and prepare new context

                if saved_context is None:
                    # No transition happened, store the full sequence
                    generated_sequences.append(generated)
                    print(f'End. Generated (1) = {generated}')
                    break  # End generation for this input

                # Prepare new context
                arguments_token = self.control_tokens["up_tokens"]["arguments"]
                results_token = self.control_tokens["up_tokens"]["results"]
                end_token = self.control_tokens["up_tokens"]["end"]

                # Extract new context up to and including UNIT_PROPAGATION_ARGUMENTS / ANALYZE_CONFLICT_ARGUMENTS
                arg_indices = (generated == arguments_token).nonzero(as_tuple=True)[1]
                if len(arg_indices) == 0:
                    break  # model error
                arg_index = arg_indices[-1].item()
                print(f'ARG index = {arg_index}')
                new_context = torch.cat([torch.tensor([[self.tokenizer.bos_token_id]], device=self.device), generated[:, arg_index:]], dim=1)
                print(f'New context = {new_context}')

                # Generate continuation with new context
                generated = new_context
                while True:
                    with torch.no_grad():
                        logits = self(generated)[:, -1, :]
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=-1)

                    print(f'New context next token = {next_token}')

                    if next_token.item() == stop_token or next_token.item() == end_token:
                        print(f'New context break ending with = {generated}')
                        break
                
                # Extract results between RESULTS and END
                try:
                    results_start_idx = (generated == results_token).nonzero(as_tuple=True)[1][0].item()
                    results_end_idx = (generated == end_token).nonzero(as_tuple=True)[1][0].item() + 1
                    extracted_results = generated[:, results_start_idx:results_end_idx]
                except IndexError:
                    extracted_results = torch.tensor([], device=self.device).long()  # No valid results found

                print(f'Extracted results = {extracted_results}')

                # Append results to saved context and continue generating
                generated = torch.cat([saved_context, extracted_results], dim=-1)
                print(f'Generated = {generated}')
                saved_context = None  # Reset context since it has been updated

            generated_sequences.append(generated)

        return generated_sequences

# Mock control tokens
control_tokens = {
    "solve_tokens": {"arguments": 1, "start": 2, "end": 7},
    "up_tokens": {"arguments": 3, "start": 4, "results": 5, "end": 6}
}

# Create mock model
mock_model = MockModel(control_tokens)

# Fake input sequences (simulating SAT problem tokenized sequences)
fake_inputs = [
    [0, 1, 10, 10, 10, 10, 10, 10, 10, 10, 2],  # SOLVE_ARGUMENTS ... SOLVE_START
    # [3, 10, 10, 10, 10, 10, 4],  # UP_ARGUMENTS ... UP_START
]

# Convert to tensor format
input_tensors = [torch.tensor(seq) for seq in fake_inputs]

# Run test
output = mock_model.generate_packed(input_tensors, max_length=100, stop_token=11)
print('\n\n')

# Print results
for i, seq in enumerate(output):
    print(f"Generated {i}: {seq.tolist()}")

# Assertions to ensure expected behavior
assert len(output) >= len(fake_inputs), "Each input should produce at least one output"
assert isinstance(output[0], torch.Tensor), "Each output should be a tensor"
assert output[0].shape[1] > len(fake_inputs[0]), "Generated output should extend beyond input"

# Check if context switching happened (UP or AC must be present in extended sequence)
# for i, seq in enumerate(output):
#     seq_list = seq.tolist()[0]
#     assert any(token in seq_list for token in [21, 31]), f"Generated {i} did not switch context to UP or AC"

print("✅ Test completed successfully!")
