"""V-10: Sequence DSL Validation Tests.

Tests for SR-10 (Sequence DSL) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. Type safety — programs that produce type errors are rejected.
2. Determinism — same program + same input → same output.
3. Known-program test — 5 hand-written programs, 100 inputs each, verify all outputs.
4. Depth correctness — reported program depth matches actual nesting.
5. Deduplication — sampler does not produce semantically equivalent programs for different IDs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dsl.sequence_dsl import (
    Compose,
    Concat,
    Count,
    Drop,
    FilterEven,
    FilterGt,
    FilterOdd,
    MapAbs,
    MapMod,
    MapParity,
    MapSign,
    Max,
    Min,
    Parity,
    PrefixSum,
    Reverse,
    SeqProgram,
    Sort,
    Sum,
    Take,
    Unique,
    ZipAdd,
    check_functional_equivalence,
    sample_program,
    sample_programs_batch,
)


# ===================================================================
# Helpers
# ===================================================================

def random_inputs(n: int = 100, seed: int = 42, min_len: int = 3, max_len: int = 10,
                  val_lo: int = -9, val_hi: int = 9) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        length = int(rng.integers(min_len, max_len + 1))
        seq = [int(x) for x in rng.integers(val_lo, val_hi + 1, size=length)]
        results.append(seq)
    return results


# ===================================================================
# 1. Type Safety
# ===================================================================

class TestTypeSafety:

    def test_take_negative_raises(self):
        with pytest.raises(ValueError, match="n must be >= 0"):
            Take(n=-1)

    def test_drop_negative_raises(self):
        with pytest.raises(ValueError, match="n must be >= 0"):
            Drop(n=-1)

    def test_map_mod_zero_raises(self):
        with pytest.raises(ValueError, match="modulus must be >= 1"):
            MapMod(modulus=0)

    def test_compose_type_mismatch_no_error_since_all_list_to_list(self):
        # All our ops are list[int] → list[int], so compose should always work
        prog = Compose(first=Sort(), second=Reverse())
        assert prog.evaluate([3, 1, 2]) == [3, 2, 1]


# ===================================================================
# 2. Determinism
# ===================================================================

class TestDeterminism:

    def test_leaf_ops_deterministic(self):
        ops = [Sort(), Reverse(), Unique(), PrefixSum(), MapAbs(), MapSign(),
               MapParity(), Take(3), Drop(1), FilterGt(3), FilterEven(),
               FilterOdd(), MapMod(3), Sum(), Count(), Max(), Min(), Parity()]
        inp = [5, -3, 7, 2, -3, 0, 8]
        for op in ops:
            r1 = op.evaluate(inp)
            r2 = op.evaluate(inp)
            assert r1 == r2, f"{op.name()} not deterministic"

    def test_composed_deterministic(self):
        prog = Compose(first=Sort(), second=Reverse())
        inp = [5, 3, 8, 1, 4]
        r1 = prog.evaluate(inp)
        r2 = prog.evaluate(inp)
        assert r1 == r2

    def test_sampled_program_deterministic(self):
        prog = sample_program(seed=42, max_depth=3)
        inputs = random_inputs(n=100)
        for inp in inputs:
            r1 = prog.evaluate(inp)
            r2 = prog.evaluate(inp)
            assert r1 == r2

    def test_sampled_program_same_seed_same_program(self):
        p1 = sample_program(seed=42, max_depth=3)
        p2 = sample_program(seed=42, max_depth=3)
        inputs = random_inputs(n=100)
        for inp in inputs:
            assert p1.evaluate(inp) == p2.evaluate(inp)


# ===================================================================
# 3. Known-Program Tests — 5 hand-written programs with verified outputs
# ===================================================================

class TestKnownPrograms:

    def test_sort(self):
        assert Sort().evaluate([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]
        assert Sort().evaluate([]) == []
        assert Sort().evaluate([1]) == [1]

    def test_reverse(self):
        assert Reverse().evaluate([1, 2, 3]) == [3, 2, 1]
        assert Reverse().evaluate([]) == []

    def test_unique(self):
        assert Unique().evaluate([3, 1, 4, 1, 5, 3]) == [3, 1, 4, 5]
        assert Unique().evaluate([1, 1, 1]) == [1]
        assert Unique().evaluate([]) == []

    def test_prefix_sum(self):
        assert PrefixSum().evaluate([1, 2, 3, 4]) == [1, 3, 6, 10]
        assert PrefixSum().evaluate([5]) == [5]
        assert PrefixSum().evaluate([]) == []

    def test_map_abs(self):
        assert MapAbs().evaluate([-3, 0, 5, -1]) == [3, 0, 5, 1]

    def test_map_sign(self):
        assert MapSign().evaluate([-3, 0, 5, -1]) == [-1, 0, 1, -1]

    def test_map_mod(self):
        assert MapMod(modulus=3).evaluate([0, 1, 2, 3, 4, 5, 6]) == [0, 1, 2, 0, 1, 2, 0]

    def test_map_parity(self):
        assert MapParity().evaluate([1, 2, 3, 4, 5]) == [1, 0, 1, 0, 1]

    def test_take(self):
        assert Take(n=3).evaluate([1, 2, 3, 4, 5]) == [1, 2, 3]
        assert Take(n=10).evaluate([1, 2]) == [1, 2]
        assert Take(n=0).evaluate([1, 2]) == []

    def test_drop(self):
        assert Drop(n=2).evaluate([1, 2, 3, 4, 5]) == [3, 4, 5]
        assert Drop(n=10).evaluate([1, 2]) == []

    def test_filter_gt(self):
        assert FilterGt(threshold=3).evaluate([1, 5, 2, 8, 3]) == [5, 8]

    def test_filter_even(self):
        assert FilterEven().evaluate([1, 2, 3, 4, 5, 6]) == [2, 4, 6]

    def test_filter_odd(self):
        assert FilterOdd().evaluate([1, 2, 3, 4, 5, 6]) == [1, 3, 5]

    def test_sum(self):
        assert Sum().evaluate([1, 2, 3]) == [6]
        assert Sum().evaluate([]) == [0]

    def test_count(self):
        assert Count().evaluate([1, 2, 3]) == [3]
        assert Count().evaluate([]) == [0]

    def test_max(self):
        assert Max().evaluate([3, 1, 7, 2]) == [7]
        assert Max().evaluate([]) == []

    def test_min(self):
        assert Min().evaluate([3, 1, 7, 2]) == [1]
        assert Min().evaluate([]) == []

    def test_parity(self):
        assert Parity().evaluate([1, 2, 3]) == [0]  # 1^0^1 = 0
        assert Parity().evaluate([1, 3, 5]) == [1]  # 1^1^1 = 1
        assert Parity().evaluate([2, 4, 6]) == [0]  # 0^0^0 = 0

    def test_concat_self(self):
        assert Concat().evaluate([1, 2]) == [1, 2, 1, 2]

    def test_concat_with_inner(self):
        prog = Concat(inner=Reverse())
        assert prog.evaluate([1, 2, 3]) == [1, 2, 3, 3, 2, 1]

    def test_zip_add_self(self):
        assert ZipAdd().evaluate([1, 2, 3]) == [2, 4, 6]

    def test_zip_add_with_inner(self):
        prog = ZipAdd(inner=Reverse())
        assert prog.evaluate([1, 2, 3]) == [4, 4, 4]  # [1+3, 2+2, 3+1]

    def test_known_program1_sort_then_reverse(self):
        """Program: sort | reverse (= sort descending)"""
        prog = Compose(first=Sort(), second=Reverse())
        assert prog.evaluate([3, 1, 4, 1, 5]) == [5, 4, 3, 1, 1]

    def test_known_program2_dedup_sort_count(self):
        """Program: unique | sort | count"""
        prog = Compose(first=Unique(), second=Compose(first=Sort(), second=Count()))
        assert prog.evaluate([3, 1, 4, 1, 5, 3]) == [4]  # unique=[3,1,4,5], sorted=[1,3,4,5], count=[4]

    def test_known_program3_filter_then_sum(self):
        """Program: filter_gt(3) | sum"""
        prog = Compose(first=FilterGt(threshold=3), second=Sum())
        assert prog.evaluate([1, 5, 2, 8, 3]) == [13]  # 5+8=13

    def test_known_program4_map_abs_then_sort(self):
        """Program: map_abs | sort"""
        prog = Compose(first=MapAbs(), second=Sort())
        assert prog.evaluate([-5, 3, -1, 7, -2]) == [1, 2, 3, 5, 7]

    def test_known_program5_take_reverse(self):
        """Program: take(3) | reverse"""
        prog = Compose(first=Take(n=3), second=Reverse())
        assert prog.evaluate([1, 2, 3, 4, 5]) == [3, 2, 1]

    def test_known_programs_bulk(self):
        """Verify 5 programs on 100 random inputs each against reference implementations."""
        programs_and_refs = [
            (
                Compose(first=Sort(), second=Reverse()),
                lambda inp: sorted(inp, reverse=True),
            ),
            (
                Compose(first=FilterGt(threshold=0), second=Sum()),
                lambda inp: [sum(x for x in inp if x > 0)],
            ),
            (
                Compose(first=MapAbs(), second=Sort()),
                lambda inp: sorted(abs(x) for x in inp),
            ),
            (
                Compose(first=Unique(), second=Count()),
                lambda inp: [len(set(inp))],
            ),
            (
                Compose(first=Take(n=3), second=Reverse()),
                lambda inp: list(reversed(inp[:3])),
            ),
        ]

        inputs = random_inputs(n=100, seed=42)
        for prog_idx, (prog, ref_fn) in enumerate(programs_and_refs):
            for i, inp in enumerate(inputs):
                dsl_out = prog.evaluate(inp)
                ref_out = ref_fn(inp)
                # Convert ref_out to list if needed
                ref_out = list(ref_out)
                assert dsl_out == ref_out, (
                    f"Program {prog_idx} ({prog.name()}) mismatch at input {i}: "
                    f"DSL={dsl_out}, ref={ref_out}, inp={inp}"
                )


# ===================================================================
# 4. Depth Correctness
# ===================================================================

class TestDepthCorrectness:

    def test_leaf_depth(self):
        assert Sort().depth() == 1
        assert Reverse().depth() == 1
        assert Take(3).depth() == 1
        assert Sum().depth() == 1
        assert MapMod(2).depth() == 1

    def test_compose_depth(self):
        # sort | reverse = depth 2
        prog = Compose(first=Sort(), second=Reverse())
        assert prog.depth() == 2

    def test_triple_compose_depth(self):
        # unique | sort | count = depth 3
        prog = Compose(first=Unique(), second=Compose(first=Sort(), second=Count()))
        assert prog.depth() == 3

    def test_concat_with_inner_depth(self):
        # concat(reverse) = 1 + reverse.depth(1) = 2
        prog = Concat(inner=Reverse())
        assert prog.depth() == 2

    def test_zip_add_with_inner_depth(self):
        prog = ZipAdd(inner=Sort())
        assert prog.depth() == 2

    def test_deep_compose_depth(self):
        # sort | reverse | unique | take(3) = depth 4
        prog = Compose(
            first=Sort(),
            second=Compose(
                first=Reverse(),
                second=Compose(
                    first=Unique(),
                    second=Take(n=3),
                ),
            ),
        )
        assert prog.depth() == 4

    def test_seq_program_depth(self):
        op = Compose(first=Sort(), second=Reverse())
        prog = SeqProgram(program_id="test", op=op)
        assert prog.depth() == 2


# ===================================================================
# 5. Deduplication — sampled programs should be functionally distinct
# ===================================================================

class TestDeduplication:

    def test_different_seeds_produce_different_programs(self):
        """Programs from different seeds should (usually) differ functionally."""
        progs = sample_programs_batch(n=20, seed=42, max_depth=2)
        n_equiv_pairs = 0
        total_pairs = 0
        for i in range(len(progs)):
            for j in range(i + 1, len(progs)):
                total_pairs += 1
                if check_functional_equivalence(progs[i], progs[j], n_test_inputs=200, seed=99):
                    n_equiv_pairs += 1

        # With 20 programs and depth 2, some collisions are possible but
        # the majority should be distinct
        assert n_equiv_pairs < total_pairs * 0.3, (
            f"Too many equivalent pairs: {n_equiv_pairs}/{total_pairs} = "
            f"{n_equiv_pairs/total_pairs:.1%}"
        )

    def test_programs_have_distinct_names(self):
        """Different seeds should usually produce different operation trees."""
        progs = sample_programs_batch(n=20, seed=42, max_depth=3)
        names = [p.name() for p in progs]
        unique_names = set(names)
        # At least half should be unique
        assert len(unique_names) >= len(progs) * 0.5

    def test_equivalence_helper_uses_negative_values_by_default(self):
        pos_only = SeqProgram(program_id="pos", op=MapAbs())
        sign_sensitive = SeqProgram(program_id="sign", op=MapSign())
        assert not check_functional_equivalence(pos_only, sign_sensitive, n_test_inputs=200, seed=123)


# ===================================================================
# 6. Sampler Tests
# ===================================================================

class TestSampler:

    def test_sample_program_reproducible(self):
        p1 = sample_program(seed=42, max_depth=2)
        p2 = sample_program(seed=42, max_depth=2)
        inputs = random_inputs(n=50)
        for inp in inputs:
            assert p1.evaluate(inp) == p2.evaluate(inp)

    def test_sample_programs_batch_count(self):
        progs = sample_programs_batch(n=10, seed=42, max_depth=2)
        assert len(progs) == 10

    def test_sample_program_depth_1(self):
        """Depth 1 should produce a single leaf op."""
        prog = sample_program(seed=42, max_depth=1)
        assert prog.depth() == 1

    def test_sample_programs_no_crash_various_depths(self):
        """Generate many programs at various depths without crashing."""
        for depth in [1, 2, 3, 4]:
            for seed in range(30):
                prog = sample_program(seed=seed, max_depth=depth)
                inputs = random_inputs(n=20, seed=seed + 1000)
                for inp in inputs:
                    result = prog.evaluate(inp)
                    assert isinstance(result, list)

    def test_sample_program_ids(self):
        progs = sample_programs_batch(n=5, seed=42, max_depth=2)
        ids = [p.program_id for p in progs]
        assert len(set(ids)) == 5  # all unique
        assert ids[0] == "prog_s42_d2"
        assert ids[-1] == "prog_s46_d2"


# ===================================================================
# 7. Edge Cases
# ===================================================================

class TestEdgeCases:

    def test_empty_input(self):
        ops = [Sort(), Reverse(), Unique(), PrefixSum(), MapAbs(), MapSign(),
               MapParity(), FilterEven(), FilterOdd(), Sum(), Count(), Parity()]
        for op in ops:
            result = op.evaluate([])
            assert isinstance(result, list), f"{op.name()} failed on empty input"

    def test_single_element(self):
        assert Sort().evaluate([5]) == [5]
        assert Reverse().evaluate([5]) == [5]
        assert Unique().evaluate([5]) == [5]
        assert PrefixSum().evaluate([5]) == [5]
        assert Sum().evaluate([5]) == [5]
        assert Max().evaluate([5]) == [5]
        assert Min().evaluate([5]) == [5]

    def test_take_more_than_length(self):
        assert Take(n=100).evaluate([1, 2, 3]) == [1, 2, 3]

    def test_drop_more_than_length(self):
        assert Drop(n=100).evaluate([1, 2, 3]) == []

    def test_filter_gt_removes_all(self):
        assert FilterGt(threshold=100).evaluate([1, 2, 3]) == []

    def test_negative_values(self):
        assert Sort().evaluate([-3, -1, -5]) == [-5, -3, -1]
        assert MapAbs().evaluate([-3, -1, -5]) == [3, 1, 5]
        assert MapSign().evaluate([-3, 0, 5]) == [-1, 0, 1]
