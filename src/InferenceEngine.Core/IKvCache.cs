namespace InferenceEngine.Core;

/// <summary>
/// Key/value cache for one generation session, owned by <c>InferenceEngine.Engine</c> and
/// passed into <see cref="IModel.Forward"/> so the forward pass and the cache implementation
/// stay independently swappable — see the paged/head-major kv-cache ADR
/// (<c>docs/architecture/260914-kv-cache.md</c>) for why the accessors below are shaped the way
/// they are (per-KV-head, not per-layer-row) and what block addressing buys.
/// </summary>
public interface IKvCache
{
    /// <summary>Tokens per physical block. A power of two, so callers can shift/mask instead of divide.</summary>
    int BlockSize { get; }

    /// <summary>Floats per key/value slot for one head.</summary>
    int HeadDim { get; }

    /// <summary>Maximum sequence position this cache can ever hold, fixed at construction.</summary>
    int Capacity { get; }

    /// <summary>
    /// Number of resident tokens (the highest <c>position + 1</c> passed to <see cref="Reserve"/>
    /// so far, or after a <see cref="Rollback"/>). Unlike the old contiguous cache's fixed
    /// capacity, this now tracks actual fill state, since storage is allocated lazily.
    /// </summary>
    int Length { get; }

    /// <summary>
    /// Ensures the physical block backing <paramref name="position"/> is assigned, allocating one
    /// if this is the first position to land in a not-yet-assigned logical block. Must be called
    /// before <see cref="KeySlot"/>/<see cref="ValueSlot"/> for a new position — kept as an
    /// explicit, separately-callable hook (rather than hidden inside the slot accessors) so a
    /// future copy-on-write cache can intercept allocation without changing write call sites.
    /// </summary>
    void Reserve(int position);

    /// <summary>The writable key slot (<see cref="HeadDim"/> floats) for one KV head at one position.</summary>
    Span<float> KeySlot(int layer, int kvHead, int position);

    /// <summary>The writable value slot (<see cref="HeadDim"/> floats) for one KV head at one position.</summary>
    Span<float> ValueSlot(int layer, int kvHead, int position);

    /// <summary>
    /// A contiguous <c>[count, HeadDim]</c> tile of key data for one KV head, covering positions
    /// <c>[logicalBlock * BlockSize, logicalBlock * BlockSize + count)</c>. <paramref name="count"/>
    /// must not cross a block boundary (at most <see cref="BlockSize"/>).
    /// </summary>
    ReadOnlySpan<float> KeyBlockForHead(int layer, int kvHead, int logicalBlock, int count);

    /// <summary>Same contract as <see cref="KeyBlockForHead"/>, for value data.</summary>
    ReadOnlySpan<float> ValueBlockForHead(int layer, int kvHead, int logicalBlock, int count);

    /// <summary>Releases every assigned block and resets <see cref="Length"/> to zero. Equivalent to <c>Rollback(0)</c>.</summary>
    void Reset();

    /// <summary>
    /// Releases every block that holds only positions <c>&gt;= toPosition</c> and sets
    /// <see cref="Length"/> to <paramref name="toPosition"/>. A deliberate seam for multi-turn
    /// session reuse and speculative-decoding rollback — see the kv-cache ADR's Consequences;
    /// neither this nor <see cref="Reset"/> has a production caller yet.
    /// </summary>
    void Rollback(int toPosition);
}

// Note: the cache tracks which logical blocks are assigned (needed for Reserve/Rollback to know
// what to allocate/free), but still NOT which positions within an assigned block are causally
// attendable at any given decode step — that stays the caller's job (GqaAttention walks t in
// [0, position], not [0, Length)).
