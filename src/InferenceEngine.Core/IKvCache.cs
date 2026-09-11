namespace InferenceEngine.Core;

/// <summary>
/// Key/value cache for one generation session, owned by <c>InferenceEngine.Engine</c> and
/// passed into <see cref="IModel.Forward"/> so the forward pass and the cache implementation
/// stay independently swappable (contiguous today, paged/quantized later, if at all).
/// </summary>
public interface IKvCache
{
    /// <summary>Capacity in tokens (prompt length + max new tokens), fixed at construction.</summary>
    int Length { get; }

    /// <summary>The writable key slot for <paramref name="layer"/> at position <paramref name="position"/>.</summary>
    Span<float> KeySlot(int layer, int position);

    /// <summary>The writable value slot for <paramref name="layer"/> at position <paramref name="position"/>.</summary>
    Span<float> ValueSlot(int layer, int position);

    /// <summary>The read-only key slot for <paramref name="layer"/> at position <paramref name="position"/>.</summary>
    ReadOnlySpan<float> Key(int layer, int position);

    /// <summary>The read-only value slot for <paramref name="layer"/> at position <paramref name="position"/>.</summary>
    ReadOnlySpan<float> Value(int layer, int position);
}

// Note: no "advance" / current-length method — the caller (Engine's generation loop) always
// knows and passes the absolute position, so causal attention simply reads Key/Value for
// t in [0, position] without the cache tracking any mutable fill state of its own.
