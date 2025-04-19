from typing import List, Tuple
from tokenizers import Tokenizer
from src.dataset.trace import TraceRaw, TraceTokenized


def apply_loss_mask(
    ids: List[int], 
    structural_tokens: List[int], 
    block_markers: List[Tuple[int, int]], 
    shift: bool = True
) -> List[int]:
    labels = ids[:]
    inside_block = [False] * len(block_markers)

    for i, tid in enumerate(ids):
        if tid in structural_tokens:
            labels[i] = -100
        for j, (begin_id, end_id) in enumerate(block_markers):
            if tid == begin_id:
                inside_block[j] = True
            if inside_block[j]:
                labels[i] = -100
            if tid == end_id:
                inside_block[j] = False

    return labels[1:] + [-100] if shift else labels


def tokenize_trace_batch(
    trace_examples: List[TraceRaw],
    tokenizer: Tokenizer,
    single_tokens: List[int],
    block_tokens: List[Tuple[int, int]],
) -> List[TraceTokenized]:
    
    def tokenize(text: str) -> dict:
        e = tokenizer.encode(text)
        return {
            "input_ids": e.ids,
            "attention_mask": e.attention_mask,
            "labels": apply_loss_mask(e.ids, single_tokens, block_tokens),
        }

    def tokenize_list(texts: List[str]) -> List[dict]:
        return [tokenize(t) for t in texts]

    return [
        TraceTokenized(
            input_clauses=tokenize(trace.input_clauses),
            solve=tokenize(trace.solve),
            unit_propagation=tokenize_list(trace.unit_propagation),
            analyze_conflict=tokenize_list(trace.analyze_conflict),
        )
        for trace in trace_examples
    ]

def tokenize_trace_batch_wrapper(args):
    batch, tokenizer, structural_tokens, block_markers = args
    return tokenize_trace_batch(batch, tokenizer, structural_tokens, block_markers)
