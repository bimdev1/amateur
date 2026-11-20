"""Provenance carrier draft: validation, ordering, and exact-type interaction.

These tests exercise ``amateur/provenance.py`` only; the executor builds
every envelope in ``DryRun._derive`` before arrival (no payload is minted
at emit time), so nothing here claims the physics-epoch gate is closed.
What is pinned:

- ``Provenance`` validation: ``source_fid`` is a ledger flow id (1..27),
  ``tick_born`` is a non-negative tick, the record is frozen and hashable;
- the emission invariant ``tick_born <= emit_tick``, and an explicit
  record of its limit (same-tick birth passes: ordering is a tripwire, not
  proof of derivation);
- the envelope does not weaken the typed-payload whitelist: exact-type
  membership is preserved on the body and on the envelope itself, so
  ``Sneaky(MatOverlay)`` is still rejected;
- provenance changes neither the body's exact type nor its hashability
  semantics;
- the read-continuity signature that a minted-at-emit read would break:
  a store read's provenance must equal the stored record's provenance.
"""
from __future__ import annotations

import dataclasses
import inspect
import unittest

from amateur.ledger import FLOW_LEDGER
from amateur.payloads import (
    READ_BODY,
    WRITE_BODY,
    BiasPrimitive,
    DigestedDebrisBody,
    EmergentCrystalBody,
    MatOverlay,
    PAYLOAD_SCHEMA,
)
from amateur.dryrun import DryRun
from amateur.provenance import (
    FLOW_IDS,
    STORE_READ_FIDS,
    STORE_WRITE_FIDS,
    Envelope,
    Provenance,
    allowed_body,
    allowed_envelope,
    derived_at,
    provenance_of,
    require_channel_provenance,
    require_envelope,
    require_provenance,
)


class Sneaky(MatOverlay):
    """A subclass of an allowed payload: must never pass exact-type checks."""


class Smooth(BiasPrimitive):
    """A subclass of the D4 payload: must never pass exact-type checks."""


class SneakyEnvelope(Envelope):
    """A subclass of the envelope: also rejected (the gate is exact-type)."""


def _overlay(tick: int = 1, oid: str = "cry-1") -> MatOverlay:
    return MatOverlay(tick, oid, "active")


class TestProvenanceRecord(unittest.TestCase):
    def test_flow_ids_are_exactly_the_ledger(self) -> None:
        self.assertEqual(FLOW_IDS, frozenset(range(1, 28)))
        self.assertEqual(
            FLOW_IDS, frozenset(flow.fid for flow in FLOW_LEDGER)
        )

    def test_every_ledger_flow_id_is_accepted(self) -> None:
        for fid in FLOW_IDS:
            prov = Provenance(fid, 0)
            self.assertEqual((prov.source_fid, prov.tick_born), (fid, 0))

    def test_invalid_source_fid_rejected(self) -> None:
        for bad in (0, 28, -1, 1.0, "1", None, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Provenance(bad, 0)  # type: ignore[arg-type]

    def test_invalid_tick_born_rejected(self) -> None:
        for bad in (-1, 1.5, "0", None, False):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Provenance(1, bad)  # type: ignore[arg-type]

    def test_record_is_frozen(self) -> None:
        prov = Provenance(22, 3)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            prov.tick_born = 4  # type: ignore[misc]

    def test_equality_and_hash_are_value_based(self) -> None:
        self.assertEqual(Provenance(23, 2), Provenance(23, 2))
        self.assertNotEqual(Provenance(23, 2), Provenance(24, 2))
        self.assertEqual(
            {Provenance(23, 2), Provenance(23, 2)}, {Provenance(23, 2)}
        )
        self.assertEqual(hash(Provenance(23, 2)), hash(Provenance(23, 2)))


class TestTickOrdering(unittest.TestCase):
    def test_strictly_earlier_birth_is_accepted(self) -> None:
        self.assertEqual(
            require_provenance(Provenance(23, 1), 4), Provenance(23, 1)
        )

    def test_same_tick_birth_is_accepted_for_writes(self) -> None:
        # A value produced this tick is legitimately born this tick. This
        # is exactly why ordering alone cannot prove "no minting at emit
        # time": a fabricated write can also claim tick_born == emit tick.
        self.assertEqual(
            require_provenance(Provenance(22, 7), 7), Provenance(22, 7)
        )

    def test_future_dated_birth_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_provenance(Provenance(23, 5), 4)

    def test_missing_provenance_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            require_provenance(_overlay(), 4)
        with self.assertRaises(TypeError):
            require_provenance(None, 4)

    def test_invalid_emit_tick_rejected(self) -> None:
        for bad in (-1, 1.5, "1", None):
            with self.assertRaises(ValueError):
                require_provenance(Provenance(1, 0), bad)  # type: ignore[arg-type]

    def test_derived_at_is_the_same_gate(self) -> None:
        self.assertEqual(
            derived_at(Provenance(24, 2), 3), Provenance(24, 2)
        )
        with self.assertRaises(ValueError):
            derived_at(Provenance(24, 3), 2)
        with self.assertRaises(TypeError):
            derived_at(_overlay(), 2)

    def test_envelope_and_bare_record_are_both_readable(self) -> None:
        prov = Provenance(23, 2)
        self.assertEqual(provenance_of(prov), prov)
        self.assertEqual(provenance_of(Envelope(prov, _overlay())), prov)
        self.assertEqual(derived_at(Envelope(prov, _overlay()), 3), prov)


def _write_body(fid: int) -> object:
    if fid == 18:
        return EmergentCrystalBody("cand-0", (0.0,) * 64, 0.0, 1)
    if fid == 24:
        return MatOverlay(1, "ovr-0", "active")
    if fid == 22:
        return BiasPrimitive("bias-0", 0.4)
    if fid == 19:
        from amateur.payloads import DigestedDebrisBody

        return DigestedDebrisBody("debris-0", ())
    raise AssertionError(f"no write body for #{fid}")


class TestEnvelopeWhitelist(unittest.TestCase):
    def test_exact_bodies_are_accepted(self) -> None:
        for fid in WRITE_BODY:
            body = _write_body(fid)
            env = Envelope(Provenance(fid, 1), body)
            self.assertTrue(allowed_envelope(fid, env), msg=f"#{fid}")
            self.assertIs(require_envelope(fid, env, 1), env)

    def test_read_channels_accept_none_and_their_exact_body(self) -> None:
        for fid in READ_BODY:
            env = Envelope(Provenance(fid, 1), None)
            self.assertTrue(allowed_envelope(fid, env), msg=f"#{fid}")

    def test_body_subclasses_are_still_rejected(self) -> None:
        for fid in WRITE_BODY:
            env = Envelope(
                Provenance(fid, 1), Sneaky(1, "x", "active")
            )
            self.assertFalse(allowed_envelope(fid, env), msg=f"#{fid}")
            with self.assertRaises(TypeError):
                require_envelope(fid, env, 1)
        for fid in WRITE_BODY:
            env = Envelope(Provenance(fid, 1), Smooth("x", 0.5))
            self.assertFalse(allowed_envelope(fid, env), msg=f"#{fid}")
            with self.assertRaises(TypeError):
                require_envelope(fid, env, 1)

    def test_envelope_subclass_is_rejected(self) -> None:
        env = SneakyEnvelope(Provenance(24, 1), _overlay())
        self.assertFalse(allowed_envelope(24, env))
        with self.assertRaises(TypeError):
            require_envelope(24, env, 1)

    def test_cross_channel_bodies_are_rejected(self) -> None:
        overlay_env = Envelope(Provenance(22, 1), _overlay())
        bias_env = Envelope(Provenance(23, 1), BiasPrimitive("b", 0.5))
        for fid in (22,):
            self.assertFalse(allowed_envelope(fid, overlay_env))
        for fid in (18, 24):
            self.assertFalse(allowed_envelope(fid, bias_env))

    def test_raw_payload_is_not_accepted_at_the_channel(self) -> None:
        for fid in WRITE_BODY:
            body = _overlay()
            self.assertFalse(allowed_envelope(fid, body), msg=f"#{fid}")
            with self.assertRaises(TypeError):
                require_envelope(fid, body, 1)

    def test_placeholder_channels_accept_none_with_provenance(self) -> None:
        for fid in (1, 9, 11, 26):
            env = Envelope(Provenance(fid, 2), None)
            self.assertTrue(allowed_envelope(fid, env), msg=f"#{fid}")
            self.assertEqual(require_envelope(fid, env, 2), env)
        # Read channels may be empty before their store fills.
        self.assertTrue(
            allowed_envelope(8, Envelope(Provenance(23, 0), None))
        )

    def test_write_channels_reject_an_empty_body(self) -> None:
        for fid in WRITE_BODY:
            self.assertFalse(
                allowed_envelope(fid, Envelope(Provenance(fid, 1), None))
            )

    def test_ordering_is_enforced_at_the_channel(self) -> None:
        env = Envelope(Provenance(24, 6), _overlay(6))
        with self.assertRaises(ValueError):
            require_envelope(24, env, 5)
        self.assertIs(require_envelope(24, env, 6), env)

    def test_every_flow_has_a_defined_body_whitelist(self) -> None:
        self.assertEqual(set(PAYLOAD_SCHEMA), set(FLOW_IDS))
        for fid, allowed in PAYLOAD_SCHEMA.items():
            self.assertTrue(allowed, msg=f"#{fid}")
            for body_type in allowed:
                self.assertIsInstance(body_type, type, msg=f"#{fid}")


class TestFieldStyleEmulation(unittest.TestCase):
    """The per-type-field option, emulated locally.

    It does not weaken the whitelist either, as long as membership remains
    exact-type. The cost it demonstrates is that every payload type must
    repeat the two fields and the executor cannot attach provenance to the
    placeholder channels, which carry ``None``.
    """

    def test_fields_do_not_weaken_exact_type_membership(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class FieldOverlay:
            epoch: int
            overlay_id: str
            status: str
            source_fid: int
            tick_born: int

        class SneakyField(FieldOverlay):
            pass

        schema = frozenset({FieldOverlay})
        self.assertTrue(type(FieldOverlay(1, "x", "active", 23, 1)) in schema)
        self.assertFalse(
            type(SneakyField(1, "x", "active", 23, 1)) in schema
        )

    def test_fields_are_readable_by_the_option_agnostic_accessor(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class FieldBias:
            bias_primitive: str
            confidence: float
            source_fid: int
            tick_born: int

        value = FieldBias("b", 0.5, 22, 3)
        self.assertEqual(provenance_of(value), Provenance(22, 3))
        self.assertEqual(derived_at(value, 4), Provenance(22, 3))


class TestPayloadTypeAndHashStability(unittest.TestCase):
    def test_body_exact_type_is_unchanged(self) -> None:
        body = _overlay()
        env = Envelope(Provenance(24, 1), body)
        self.assertIs(env.body, body)
        self.assertIs(type(env.body), MatOverlay)
        self.assertIs(type(env.body), type(body))

    def test_subclass_remains_a_distinct_type(self) -> None:
        sneaky = Sneaky(1, "x", "active")
        self.assertIsNot(type(sneaky), MatOverlay)
        self.assertIsInstance(sneaky, MatOverlay)
        self.assertNotEqual(sneaky, _overlay(1, "x"))
        self.assertNotEqual(_overlay(1, "x"), sneaky)

    def test_hashing_follows_the_body(self) -> None:
        body = _overlay()
        env = Envelope(Provenance(24, 1), body)
        self.assertEqual(hash(env.body), hash(MatOverlay(1, "cry-1", "active")))
        self.assertEqual(hash(env), hash(Envelope(Provenance(24, 1), body)))
        self.assertNotEqual(
            Envelope(Provenance(24, 1), body),
            Envelope(Provenance(25, 1), body),
        )

    def test_unhashable_body_is_not_made_hashable(self) -> None:
        env = Envelope(Provenance(1, 1), {"k": 1})
        with self.assertRaises(TypeError):
            hash(env)

    def test_body_equality_is_unaffected_by_provenance(self) -> None:
        left = MatOverlay(1, "ovr-0", "active")
        right = MatOverlay(1, "ovr-0", "active")
        self.assertEqual(left, right)
        self.assertEqual(
            Envelope(Provenance(23, 1), left),
            Envelope(Provenance(23, 1), right),
        )
        self.assertNotEqual(
            Envelope(Provenance(23, 1), left),
            Envelope(Provenance(24, 1), right),
        )


class TestNoMintingTripwire(unittest.TestCase):
    """What the ordering check can and cannot prove.

    A read of stored state must pre-date its emission and must carry the
    stored record's provenance; a value minted at emit time would carry
    ``(read_fid, emit_tick)`` and fail the continuity equality even though
    it passes the ordering check. This is the signature the integration
    test must assert.
    """

    def test_ordering_alone_accepts_same_tick_minting(self) -> None:
        minted = Provenance(24, 5)
        self.assertEqual(require_provenance(minted, 5), minted)
        # Hence the memo's structural + continuity proof obligations.

    def test_read_inherits_the_stored_provenance(self) -> None:
        stored = Envelope(
            Provenance(24, 3), MatOverlay(3, "ovr-0", "active")
        )
        read = Envelope(
            Provenance(24, 3), MatOverlay(3, "ovr-0", "stale")
        )
        self.assertEqual(derived_at(read, 5), stored.provenance)
        self.assertLess(read.provenance.tick_born, 5)

    def test_reminted_read_fails_continuity(self) -> None:
        stored = Envelope(
            Provenance(24, 3), MatOverlay(3, "ovr-0", "active")
        )
        reminted = Envelope(
            Provenance(25, 5), MatOverlay(3, "ovr-0", "stale")
        )
        self.assertEqual(derived_at(reminted, 5), Provenance(25, 5))
        self.assertNotEqual(reminted.provenance, stored.provenance)

    def test_write_is_born_at_its_emission_tick(self) -> None:
        write = Envelope(
            Provenance(24, 5), MatOverlay(5, "ovr-0", "active")
        )
        self.assertEqual(
            require_provenance(write, 5), Provenance(24, 5)
        )


class TestChannelProvenanceRule(unittest.TestCase):
    """The adjudicated per-flow birth rule on top of the envelope gate."""

    def test_store_read_requires_a_strictly_prior_birth(self) -> None:
        same_tick = Envelope(
            Provenance(23, 5), MatOverlay(5, "cry-5", "stale")
        )
        with self.assertRaises(ValueError):
            require_channel_provenance(25, same_tick, 5)
        earlier = Envelope(
            Provenance(23, 4), MatOverlay(4, "cry-4", "stale")
        )
        self.assertIs(require_channel_provenance(25, earlier, 5), earlier)

    def test_write_requires_same_tick_birth(self) -> None:
        late = Envelope(Provenance(24, 4), MatOverlay(4, "ovr-0", "active"))
        with self.assertRaises(ValueError):
            require_channel_provenance(24, late, 5)
        born_now = Envelope(
            Provenance(24, 5), MatOverlay(5, "ovr-0", "active")
        )
        self.assertIs(require_channel_provenance(24, born_now, 5), born_now)

    def test_placeholder_and_empty_read_require_same_tick(self) -> None:
        for fid in (1, 7, 11):
            born_now = Envelope(Provenance(fid, 2), None)
            self.assertIs(
                require_channel_provenance(fid, born_now, 2), born_now
            )
            stale = Envelope(Provenance(fid, 1), None)
            with self.assertRaises(ValueError):
                require_channel_provenance(fid, stale, 2)

    def test_channel_gate_keeps_the_exact_type_whitelist(self) -> None:
        with self.assertRaises(TypeError):
            require_channel_provenance(24, _overlay(), 1)
        sneaky = Envelope(Provenance(24, 1), Sneaky(1, "x", "active"))
        with self.assertRaises(TypeError):
            require_channel_provenance(24, sneaky, 1)

    def test_read_write_sets_are_derived_from_the_ledger(self) -> None:
        self.assertEqual(
            STORE_READ_FIDS,
            frozenset(f.fid for f in FLOW_LEDGER if "(read)" in f.flow),
        )
        self.assertEqual(
            STORE_WRITE_FIDS,
            frozenset(
                f.fid for f in FLOW_LEDGER if "(read)" not in f.flow
                and f.target.startswith(("D1", "D2", "D3", "D4"))
            ),
        )


class TestExecutorContinuity(unittest.TestCase):
    """End-to-end provenance continuity over a real run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dry = DryRun()
        cls.dry.run(12)
        cls.events = cls.dry.emissions

    def test_reads_are_born_before_they_are_emitted(self) -> None:
        for tick, fid, envelope in self.events:
            born = envelope.provenance.tick_born
            if fid in STORE_READ_FIDS and envelope.body is not None:
                self.assertLess(born, tick, msg=f"tick {tick} read #{fid}")
            else:
                self.assertEqual(born, tick, msg=f"tick {tick} flow #{fid}")

    def test_read_provenance_names_a_prior_write(self) -> None:
        by_birth = {
            (envelope.provenance.source_fid,
             envelope.provenance.tick_born): envelope
            for _tick, fid, envelope in self.events
            if fid not in STORE_READ_FIDS
        }
        all_cohort_ids: set[str] = set()
        for _tick, fid, envelope in self.events:
            if fid == 19 and envelope.body is not None:
                all_cohort_ids.update(
                    cohort.cohort_id for cohort in envelope.body.cohorts
                )
        for tick, fid, envelope in self.events:
            if fid not in STORE_READ_FIDS or envelope.body is None:
                continue
            provenance = envelope.provenance
            if provenance == Provenance(23, 0):
                self.assertEqual(tick, 1, msg="#23,0 is the seed only")
                continue
            producer = by_birth[
                (provenance.source_fid, provenance.tick_born)
            ]
            if fid == 7:
                self.assertIs(envelope.body, producer.body)
            elif fid in (8, 25):
                self.assertEqual(
                    producer.body.overlay_id,
                    envelope.body.overlay_id,
                    msg=f"tick {tick} read #{fid}",
                )
            elif fid in (20, 26):
                read_ids = {
                    cohort.cohort_id for cohort in envelope.body.cohorts
                }
                self.assertTrue(
                    read_ids <= all_cohort_ids,
                    msg=f"tick {tick} read #{fid}",
                )
                self.assertIn(
                    (provenance.source_fid, provenance.tick_born), by_birth
                )
            elif fid == 21:
                self.assertIn(
                    envelope.body.motif_id, all_cohort_ids,
                    msg=f"tick {tick}",
                )
            elif fid == 23:
                self.assertEqual(
                    producer.body.candidate_id,
                    envelope.body.candidate_id,
                    msg=f"tick {tick}",
                )
            elif fid == 27:
                self.assertTrue(
                    set(envelope.body.cohort_ids) <= all_cohort_ids,
                    msg=f"tick {tick}",
                )

    def test_read_provenance_equals_the_tracked_head_provenance(self) -> None:
        run = DryRun()
        for tick in range(1, 7):
            head_id = run.board.head()[0]
            expected = run._head_provenance[head_id]
            run.tick()
            emitted = {
                fid: envelope for at, fid, envelope in run.emissions
                if at == tick
            }
            for fid in (8, 25):
                self.assertEqual(emitted[fid].provenance, expected)
                self.assertEqual(emitted[fid].body.overlay_id, head_id)

    def test_stale_read_pairs_with_the_same_tick_ratchet_write(self) -> None:
        by_tick: dict[int, dict[int, Envelope]] = {}
        for tick, fid, envelope in self.events:
            by_tick.setdefault(tick, {})[fid] = envelope
        for tick, envelopes in by_tick.items():
            self.assertEqual(
                envelopes[25].body.overlay_id,
                envelopes[24].body.overlay_id,
                msg=f"tick {tick}",
            )

    def test_seed_is_attributed_to_23_at_tick_zero(self) -> None:
        first = {fid: envelope for tick, fid, envelope in self.events
                 if tick == 1}
        for fid in (8, 25):
            self.assertEqual(first[fid].provenance, Provenance(23, 0))
            expected_status = "active" if fid == 8 else "stale"
            self.assertEqual(
                first[fid].body,
                MatOverlay(0, "ovr-0", expected_status),
            )

    def test_d4_read_hands_back_the_exact_sedimented_object(self) -> None:
        reads = {tick: envelope for tick, fid, envelope in self.events
                 if fid == 7}
        writes = {tick: envelope for tick, fid, envelope in self.events
                  if fid == 22}
        for tick in range(2, 13):
            self.assertIs(reads[tick].body, writes[tick - 1].body)
            self.assertEqual(
                reads[tick].provenance, writes[tick - 1].provenance
            )


class TestStructuralAntiMint(unittest.TestCase):
    """Structural evidence, NOT proof of derivation.

    These tests show that the emit boundary no longer constructs bodies and
    that the derivation pass runs first. They cannot prove that `_derive`
    reads real state -- that is the continuity and counterfactual tests'
    job, and the gate stays PARTIAL while #22/#23 bodies are synthetic.
    """

    def test_payload_for_is_gone(self) -> None:
        import amateur.dryrun as dryrun

        self.assertFalse(hasattr(dryrun, "_payload_for"))
        self.assertFalse(hasattr(DryRun, "_payload_for"))

    def test_arrive_constructs_no_body_and_no_provenance(self) -> None:
        source = inspect.getsource(DryRun._arrive)
        for constructor in ("MatOverlay(", "BiasPrimitive(", "Provenance("):
            self.assertNotIn(constructor, source)
        self.assertIn("self.derived[fid]", source)

    def test_derive_runs_before_any_arrival(self) -> None:
        source = inspect.getsource(DryRun.tick)
        self.assertIn("self._derive()", source)
        self.assertLess(
            source.index("self._derive()"), source.index("_arrive(")
        )

    def test_derive_fills_all_twenty_seven_flows(self) -> None:
        run = DryRun()
        run.tick()
        self.assertEqual(set(run.derived), set(FLOW_IDS))
        for envelope in run.derived.values():
            self.assertIs(type(envelope), Envelope)
            self.assertIs(type(envelope.provenance), Provenance)


if __name__ == "__main__":
    unittest.main()
