import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from flash_attn.utils.distributed import all_gather

RING_ATTN_GROUP = None


def set_ring_attn_group(group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group


def get_ring_attn_group():
    return RING_ATTN_GROUP


def reset_ring_attn_position_ids(start, end, packed_seq_lens):
    """
    Calculate position ids for packed_seq_ids[start:end].
    For example, if the packed_seq_lens is [3, 2, 4, 1], start=2, end=8,
    the position ids will be [2, 0, 1, 0, 1, 2].

    Args:
        start: the start position
        end: the end position
        packed_seq_lens: the sequence lengths of packed sequences
    """
    position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
    offset = 0
    for seqlen in packed_seq_lens:
        seq_start = max(offset, start)
        seq_end = min(offset + seqlen, end)
        if seq_start < seq_end:
            position_ids[0, seq_start - start : seq_end - start] = torch.arange(seq_start - offset, seq_end - offset)

        offset += seqlen
        if offset >= end:
            break
    return position_ids


def update_ring_attn_params(packed_seq_lens, total_seq_len):
    """
    Calculate the cu_seqlens for the current forward pass and pass the value to
    the substituted ring_flash_attn.

    Note that total_seq_len may be larger than the sum of packed_seq_lens because of padding.
    """
    assert RING_ATTN_GROUP is not None
    cu_seqlens = torch.cumsum(
        torch.tensor(packed_seq_lens, device=torch.cuda.current_device(), dtype=torch.int32),
        dim=-1,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(F.pad(cu_seqlens, (1, 0), value=0), (0, 1), value=total_seq_len)

    from ring_flash_attn import update_ring_flash_attn_params

    update_ring_flash_attn_params(cu_seqlens, RING_ATTN_GROUP)


def convert_ring_attn_params(sequences, attention_mask, packed_seq_lens, ring_attn_group, inputs_embeds, position_ids):
    # each rank within the ring group will process sequences[start:end]
    ring_attn_rank = dist.get_rank(group=ring_attn_group)
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    total_seq_len = sequences.numel()
    local_seq_len = total_seq_len // ring_attn_size
    start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
    sequences = sequences[:, start:end]
    inputs_embeds = inputs_embeds[:, start:end]
    attention_mask = attention_mask[:, start:end]
    position_ids = position_ids[..., start:end] #qwen2_5_vl has position_ids shape: [3,bs,seq_len]
    hacked_position_ids = reset_ring_attn_position_ids(start, end, packed_seq_lens)
    update_ring_attn_params(packed_seq_lens, total_seq_len)
    return sequences, attention_mask, hacked_position_ids, inputs_embeds, position_ids


def pad_sequences(sequences, attention_mask, num_actions, packed_seq_lens, ring_attn_group, pad_token_id=0):
    # Pads the input sequences and attention mask to ensure that their lengths are multiples of the ring attention size.
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    if isinstance(sequences, torch.Tensor):
        seqlen = sequences.shape[-1]
        pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
        padded = torch.tensor([pad_token_id] * pad_len, device=sequences.device, dtype=sequences.dtype).unsqueeze(0)
        sequences = torch.cat([sequences, padded], dim=1)
        attention_mask = torch.cat(
            [attention_mask, (len(sequences) + 1) * torch.ones(1, pad_len, device="cuda", dtype=torch.float)], dim=-1
        )
    elif isinstance(sequences, list):
        seqlen = len(sequences)
        pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
        sequences += [pad_token_id] * pad_len
        attention_mask += [len(packed_seq_lens) + 1] * pad_len
    else:
        raise "sequences is not available type"
    num_actions[-1] += pad_len
    packed_seq_lens[-1] += pad_len
    return pad_len, sequences, attention_mask, num_actions, packed_seq_lens


def unpad_sequences(
    pad_len,
    sequences,
    attention_mask,
    num_actions,
    packed_seq_lens,
    ring_attn_group,
    action_log_probs=None,
    values=None,
    kl=None,
):
    # Removes the padding from the input sequences, attention mask, and other optional tensors after padding.
    if pad_len > 0:
        sequences = sequences[:, :-pad_len]
        attention_mask = attention_mask[:, :-pad_len]
        num_actions[-1] -= pad_len
        packed_seq_lens[-1] -= pad_len
        if action_log_probs is not None:
            action_log_probs = action_log_probs[:, :-pad_len]
        if values is not None:
            values = values[:, :-pad_len]
        if kl is not None:
            kl = kl[:, :-pad_len]
    return sequences, attention_mask, num_actions, packed_seq_lens, action_log_probs, values, kl

HACKED_POSITION_IDS = None

#Both ring and our hack substitute flash_attn. This func must be called after ring's substitue_hf_flash_attn.
def substitute_ring_flash_attn():
    raw_flash_attention_forward = transformers.modeling_flash_attention_utils._flash_attention_forward
    def _hacked_flash_attention_forward(*args,**kwargs):
        global HACKED_POSITION_IDS
        if HACKED_POSITION_IDS is not None:
            kwargs['position_ids'] = HACKED_POSITION_IDS
        return raw_flash_attention_forward(*args,**kwargs)

    transformers.modeling_flash_attention_utils._flash_attention_forward = _hacked_flash_attention_forward

def set_hacked_position_ids(position_ids):
    global HACKED_POSITION_IDS
    HACKED_POSITION_IDS = position_ids

def clear_hacked_position_ids():
    global HACKED_POSITION_IDS
    HACKED_POSITION_IDS = None

def get_hacked_position_ids():
    global HACKED_POSITION_IDS
    return HACKED_POSITION_IDS



def get_tensor_in_current_ring_attn_rank(tensors: list[torch.Tensor] | torch.Tensor, ring_attn_group, pad_id):
    """
    Deal with padding and slice the tensor to current ring_attn_rank.
    Args:
        tensors: Each tensor shaped (batch, seqlen) or (1, total_seqs)
        ring_attn_group: Ring attention group
        pad_id: Padding id
    Returns:
        Processed tensor
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    ring_attn_rank = dist.get_rank(group=ring_attn_group)
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    seqlen = tensors[0].shape[-1]
    total_seq_len = tensors[0].numel()
    ring_attn_pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
    output_tensors = []
    for tensor in tensors:
        if tensor.numel() != total_seq_len:
            raise ValueError(f"tensor.numel() {tensor.numel()} != total_seq_len {total_seq_len}")
        tensor = torch.nn.functional.pad(tensor, (0, ring_attn_pad_len), value=pad_id)
        local_seq_len = tensor.numel() // ring_attn_size
        start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
        tensor = tensor[:, start:end]
        output_tensors.append(tensor)
    if len(output_tensors) == 1:
        output_tensors = output_tensors[0]
    return output_tensors, ring_attn_pad_len

def unpad_and_slice_tensor(sequences, attention_mask, ring_attn_group):
    """
    Unpad and slice tensor for distributed training with ring attention.

    This function performs several operations:
    1. Removes padding, unpads sequences from (batch, seqlen) to (1, total_seqs)
    2. Adapts to ring_attn_group, pads sequences to be divisible by ring_attn_group
    3. Slices the sequences for the current ring_attn_rank

    Example:
        >>> # Input sequences shape: (batch=2, seqlen=4)
        >>> sequences = [[1, 2, 3, 0], [4, 5, 0, 0]]  # 0 is padding
        >>> attention_mask = [[1, 1, 1, 0], [1, 1, 0, 0]]
        >>> # After unpad:
        >>> # sequences: [1, 2, 3, 4, 5]  # shape (1, total_seqs=5)
        >>> # If ring_attn_group size is 2, it will pad to length 6
        >>> # Then slice for current rank (e.g., rank 0 gets [1,2,3], rank 1 gets [4,5,0])

    Args:
        sequences: Input sequences tensor of shape (batch, seqlen)
        attention_mask: Attention mask tensor for the sequences
        ring_attn_group: Ring attention group for distributed processing

    Returns:
        tuple: Processed sequences and related tensors for ring attention
    """
    rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
    sequences, indices, cu_seqlens, _, _ = unpad_input(sequences.unsqueeze(-1), attention_mask)
    sequences = sequences.transpose(0, 1)  # (1, total_seqs)
    rolled_sequences = index_first_axis(
        rearrange(rolled_sequences.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(
        0, 1
    )  # (1, total_seqs)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    position_ids = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(
        0, 1
    )  # (1, total_seqs)
    ring_attn_pad_len = 0
    if ring_attn_group is not None:
        (sequences, position_ids, rolled_sequences), ring_attn_pad_len = get_tensor_in_current_ring_attn_rank(
            [sequences, position_ids, rolled_sequences], ring_attn_group, 0
        )
        cu_seqlens[-1] += ring_attn_pad_len
        update_ring_attn_params(cu_seqlens)
    return sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices


def gather_and_pad_tensor(tensor, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen):
    """
    Gather and pad tensor data (such as logits, log_probs, etc.).

    Example:
        >>> # Input tensor from each rank (shape: (1, local_seq_len))
        >>> # Rank 0: [1, 2, 3]
        >>> # Rank 1: [4, 5, 0]  # 0 is padding
        >>> # After all_gather:
        >>> # tensor: [1, 2, 3, 4, 5, 0]  # shape (1, total_seqs=6)
        >>> # After removing padding (ring_attn_pad_len=1):
        >>> # tensor: [1, 2, 3, 4, 5]  # shape (1, total_seqs=5)
        >>> # After pad_input with original indices:
        >>> # tensor: [[1, 2, 3, 0], [4, 5, 0, 0]]  # shape (batch=2, seqlen=4)

    Args:
        tensor: Input tensor, can be logits, log_probs, etc.
        ring_attn_group: Ring attention group
        ring_attn_pad_len: Padding length
        indices: Indices
        batch: Batch size
        seqlen: Sequence length

    Returns:
        Padded tensor
    """
    if ring_attn_group is not None:
        tensor = all_gather(tensor.transpose(0, 1), ring_attn_group).transpose(0, 1)  # (1, total_seqs)
        if ring_attn_pad_len > 0:
            tensor = tensor[:, :-ring_attn_pad_len]
    tensor = pad_input(tensor.transpose(0, 1), indices, batch, seqlen).squeeze(-1)  # (batch, seqlen)
    return tensor