import math

from verify_embeddings import cosine, COSINE_THRESHOLD


def test_cosine_identical_vectors_is_one():
    v = [0.1, 0.2, 0.3]
    assert abs(cosine(v, v) - 1.0) < 1e-12


def test_cosine_orthogonal_is_zero():
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-12


def test_cosine_opposite_is_minus_one():
    assert abs(cosine([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-12


def test_cosine_zero_vector_does_not_divide_by_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_float32_round_trip_stays_above_threshold():
    """The stored vectors are float32. That quantisation alone must not push a
    genuine match below the pass threshold, or the script cries wolf."""
    import struct
    v = [math.sin(i) for i in range(1536)]
    quantised = list(struct.unpack("1536f", struct.pack("1536f", *v)))
    assert cosine(v, quantised) >= COSINE_THRESHOLD


def test_threshold_rejects_a_genuinely_different_vector():
    a = [math.sin(i) for i in range(1536)]
    b = [math.cos(i) for i in range(1536)]
    assert cosine(a, b) < COSINE_THRESHOLD
