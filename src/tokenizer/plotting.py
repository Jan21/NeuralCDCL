import matplotlib.pyplot as plt

def plot_token_length_histogram(tokenized_dataset, split="train", filename="token_length.png"):
    token_lengths = [len(seq) for seq in tokenized_dataset[split]["input_ids_unpadded"]]
    plt.figure()
    plt.hist(token_lengths, bins=30, edgecolor="black")
    plt.title("Token Length Distribution")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(filename)
    print(f"Saved token length histogram: {filename}")