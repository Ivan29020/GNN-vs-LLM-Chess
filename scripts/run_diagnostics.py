"""Audit riproducibile di dati, linee scacchistiche, CSV LLM e ablation GNN.

Esecuzione dalla root del progetto:
  python -u -m scripts.run_diagnostics
  python -u -m scripts.run_diagnostics --full-dataset
  python -u -m scripts.run_diagnostics --section data|llm|gnn|ablation|all

Il programma e' di sola lettura rispetto a dataset/checkpoint/risultati esistenti.
Scrive i report esclusivamente in results/diagnostics/.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

try:
    import chess
except ImportError as exc:  # pragma: no cover - dipende dall'ambiente Colab
    raise SystemExit(
        "Dipendenza mancante: python-chess. Esegui: pip install python-chess"
    ) from exc

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover
    binomtest = None

ROOT = Path(config.BASE_DIR)
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "diagnostics"

SPLITS = {
    "train": Path(config.TRAIN_PARQUET),
    "validation": Path(config.VAL_PARQUET),
    "test": Path(config.TEST_PARQUET),
}
HELD = {
    "mixed": Path(config.HELD_OUT_PARQUET),
    "lichess_only": Path(config.HELD_OUT_LICHESS_PARQUET),
}
LLM_FILES = {
    "mixed": {
        "raw": Path(config.LLM_RAW_CSV),
        "sample": Path(config.LLM_EVAL_SAMPLE_CSV),
        "paired": Path(config.PAIRED_RESULTS_CSV),
    },
    "lichess_only": {
        "raw": Path(config.LLM_RAW_LICHESS_CSV),
        "sample": Path(config.LLM_EVAL_SAMPLE_LICHESS_CSV),
        "paired": Path(config.PAIRED_RESULTS_LICHESS_CSV),
    },
}
GNN_FILES = {
    "timed": Path(config.STAGED_SEQ_TIMED_CSV),
    "no_timing": Path(config.STAGED_SEQ_NO_TIMING_CSV),
}


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _tokens(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return str(value).strip().split()


def _norm_fen(fen: str) -> str:
    return " ".join(str(fen).split()[:3])


def _has_opening(value: Any, source: str | None = None) -> bool:
    if value is not None and not pd.isna(value):
        return _bool(value)
    return str(source or "").strip().lower() == "lichess"


def _replay(fen: str, tokens: list[str]) -> dict[str, Any]:
    """Replay UCI; distingue FEN illegale, mossa illegale e matto finale."""
    rec: dict[str, Any] = {
        "fen_parse_ok": False, "fen_board_valid": False,
        "all_moves_legal": False, "first_illegal_ply": None,
        "illegal_token": "", "plies_replayed": 0,
        "ends_in_checkmate": False, "final_fen": "", "status": "",
    }
    try:
        board = chess.Board(str(fen))
        rec["fen_parse_ok"] = True
        rec["fen_board_valid"] = bool(board.is_valid())
    except Exception as exc:
        rec["status"] = f"invalid_fen:{type(exc).__name__}"
        return rec

    for ply, token in enumerate(tokens, start=1):
        try:
            move = chess.Move.from_uci(str(token))
        except Exception:
            rec.update(status="invalid_uci", first_illegal_ply=ply, illegal_token=str(token))
            rec["plies_replayed"] = ply - 1
            rec["final_fen"] = board.fen()
            return rec
        if move not in board.legal_moves:
            rec.update(status="illegal_move", first_illegal_ply=ply, illegal_token=str(token))
            rec["plies_replayed"] = ply - 1
            rec["final_fen"] = board.fen()
            return rec
        board.push(move)
        rec["plies_replayed"] = ply

    rec["all_moves_legal"] = True
    rec["ends_in_checkmate"] = bool(board.is_checkmate())
    rec["final_fen"] = board.fen()
    rec["status"] = "checkmate" if board.is_checkmate() else ("legal_not_mate" if tokens else "empty_line")
    return rec


def _solver_move_count(tokens: list[str], has_opening: bool) -> int:
    # Lichess: index 0 e' la mossa di apertura, il solver muove agli indici dispari.
    # Polgar: FEN gia' di soluzione, solver muove agli indici pari.
    return len(tokens[1::2]) if has_opening else len(tokens[0::2])


def _solve_fen(fen: str, tokens: list[str], has_opening: bool) -> str | None:
    try:
        board = chess.Board(str(fen))
        if has_opening and tokens:
            move = chess.Move.from_uci(tokens[0])
            if move not in board.legal_moves:
                return None
            board.push(move)
        return board.fen()
    except Exception:
        return None


def _load_parquet(path: Path, cols: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        if cols is not None:
            try:
                import pyarrow.parquet as pq
                available = set(pq.ParquetFile(path).schema.names)
                cols = [c for c in cols if c in available]
            except Exception:
                pass
        return pd.read_parquet(path, columns=cols)
    except Exception as exc:
        print(f"[WARN] Non riesco a leggere {path}: {exc}")
        return None


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Non riesco a leggere {path}: {exc}")
        return None


def _write_csv(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def _add_text(lines: list[str], text: str = "") -> None:
    lines.append(text)


def _audit_dataset_frame(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    needed = {"PuzzleId", "FEN", "Moves", "mate_n"}
    missing = needed - set(df.columns)
    if missing:
        return pd.DataFrame([{"dataset": dataset_name, "error": f"colonne mancanti: {sorted(missing)}"}])

    for row in df.itertuples(index=False):
        d = row._asdict()
        source = str(d.get("source", ""))
        opening = _has_opening(d.get("has_opening_move"), source)
        fen = str(d.get("FEN", ""))
        tokens = _tokens(d.get("Moves"))
        replay = _replay(fen, tokens)
        actual_n = _solver_move_count(tokens, opening)
        solve_fen = _solve_fen(fen, tokens, opening)
        rows.append({
            "dataset": dataset_name,
            "PuzzleId": str(d.get("PuzzleId", "")),
            "source": source,
            "mate_n": int(d.get("mate_n")) if pd.notna(d.get("mate_n")) else None,
            "has_opening_move": opening,
            "reference_FEN": fen,
            "reference_Moves_UCI": " ".join(tokens),
            "target_plies": len(tokens),
            "actual_solver_moves_in_reference": actual_n,
            "line_length_matches_mate_n": actual_n == int(d.get("mate_n")) if pd.notna(d.get("mate_n")) else False,
            "original_fen_norm": _norm_fen(fen),
            "solve_fen_norm": _norm_fen(solve_fen) if solve_fen else "",
            **replay,
        })
    return pd.DataFrame(rows)


def section_data(lines: list[str], full_dataset: bool) -> None:
    print("\n[1/5] AUDIT DATASET / HELD-OUT", flush=True)
    reports = []
    source_frames: dict[str, pd.DataFrame] = {}
    for name, path in HELD.items():
        df = _load_parquet(path)
        if df is None:
            lines.append(f"DATA {name}: MANCANTE ({path})")
            print(f"  [SKIP] held-out {name}: file mancante")
            continue
        source_frames[name] = df
        audit = _audit_dataset_frame(df, f"heldout_{name}")
        reports.append(audit)
        good = audit.get("all_moves_legal", pd.Series(dtype=bool)).eq(True).sum()
        mate = audit.get("ends_in_checkmate", pd.Series(dtype=bool)).eq(True).sum()
        length_ok = audit.get("line_length_matches_mate_n", pd.Series(dtype=bool)).eq(True).sum()
        lines.append(f"HELD-OUT {name}: {len(df)} righe; linee legali {good}/{len(df)}; termina in matto {mate}/{len(df)}; lunghezza coerente con mate_n {length_ok}/{len(df)}")
        print(f"  {name}: n={len(df)}, legali={good}/{len(df)}, matto={mate}/{len(df)}, lunghezza N coerente={length_ok}/{len(df)}", flush=True)
        if "source" in df.columns:
            counts = df.groupby(["mate_n", "source"]).size()
            lines.append(f"  composizione: {counts.to_dict()}")
        if "mate_n" in audit.columns:
            bad = audit[~audit["line_length_matches_mate_n"]]
            if len(bad):
                lines.append(f"  WARNING: {len(bad)} righe hanno numero mosse del solver diverso da mate_n; vedere data_line_validation.csv")

    if reports:
        report = pd.concat(reports, ignore_index=True)
        _write_csv(report, "data_line_validation.csv")

    # Audit source sample files (300 rows total across both sample manifests).
    sample_paths = {
        "sample_mixed": Path(config.LLM_EVAL_SAMPLE_CSV),
        "sample_lichess_only": Path(config.LLM_EVAL_SAMPLE_LICHESS_CSV),
    }
    sample_reports = []
    for name, path in sample_paths.items():
        df = _load_csv(path)
        if df is None:
            lines.append(f"SAMPLE {name}: MANCANTE")
            continue
        # LLM sample stores the original FEN/Moves. Preserve the has_opening_move if present.
        if not {"PuzzleId", "FEN", "Moves", "mate_n"}.issubset(df.columns):
            lines.append(f"SAMPLE {name}: colonne non sufficienti per replay completo")
            continue
        sample_reports.append(_audit_dataset_frame(df, name))
    if sample_reports:
        sr = pd.concat(sample_reports, ignore_index=True)
        _write_csv(sr, "source_sample_line_validation.csv")
        for name, group in sr.groupby("dataset"):
            bad_len = int((~group["line_length_matches_mate_n"].fillna(False)).sum())
            bad_legal = int((~group["all_moves_legal"].fillna(False)).sum())
            bad_mate = int((~group["ends_in_checkmate"].fillna(False)).sum())
            lines.append(f"{name}: {len(group)} righe; non legali={bad_legal}; senza matto finale={bad_mate}; lunghezza target N non coerente={bad_len}")

    # Basic overlap check (IDs and normalized original FENs) for train/val/test.
    held_ids, held_fens = set(), set()
    for name, df in source_frames.items():
        if "PuzzleId" in df:
            held_ids.update(df["PuzzleId"].astype(str))
        if "FEN" in df:
            held_fens.update(df["FEN"].astype(str).map(_norm_fen))
    split_reports = []
    split_quick_frames: dict[str, pd.DataFrame] = {}
    rating_distribution_rows = []
    n_distribution_rows = []
    for split_name, path in SPLITS.items():
        # Train/validation/test Parquets do not necessarily contain the `source` column.
        cols = ["PuzzleId", "FEN", "Moves", "mate_n", "has_opening_move", "rating_bin"]
        # Quick mode reads only IDs, FEN and stratification fields; full mode replays all lines.
        df = _load_parquet(path, cols if full_dataset else ["PuzzleId", "FEN", "mate_n", "rating_bin"])
        if df is None:
            lines.append(f"SPLIT {split_name}: MANCANTE")
            continue
        id_overlap = int(df["PuzzleId"].astype(str).isin(held_ids).sum()) if "PuzzleId" in df else -1
        fen_norm = df["FEN"].astype(str).map(_norm_fen) if "FEN" in df else pd.Series(dtype=str)
        fen_overlap = int(fen_norm.isin(held_fens).sum()) if len(fen_norm) else -1
        lines.append(f"LEAKAGE {split_name} (controllo rapido ID/FEN originale): ID overlap={id_overlap}, FEN overlap={fen_overlap}")
        split_row = {"split": split_name, "rows": len(df), "id_overlap_heldout": id_overlap, "original_fen_overlap_heldout": fen_overlap}
        split_quick_frames[split_name] = df[[c for c in ["PuzzleId", "FEN"] if c in df.columns]].copy()
        if "mate_n" in df.columns:
            for n, count in df.groupby("mate_n").size().items():
                n_distribution_rows.append({"split": split_name, "mate_n": int(n), "count": int(count)})
            if "rating_bin" in df.columns:
                for (n, rb), count in df.groupby(["mate_n", "rating_bin"], dropna=False).size().items():
                    rating_distribution_rows.append({"split": split_name, "mate_n": int(n), "rating_bin": rb, "count": int(count)})
        if full_dataset and {"Moves", "mate_n"}.issubset(df.columns):
            audit = _audit_dataset_frame(df, split_name)
            split_row["illegal_lines"] = int((~audit["all_moves_legal"].fillna(False)).sum())
            split_row["not_ending_mate"] = int((~audit["ends_in_checkmate"].fillna(False)).sum())
            split_row["mate_n_mismatch"] = int((~audit["line_length_matches_mate_n"].fillna(False)).sum())
            split_positions = set(audit["solve_fen_norm"].dropna().astype(str)) | set(audit["original_fen_norm"].dropna().astype(str))
            held_positions = set()
            for held_audit in reports:
                if "solve_fen_norm" in held_audit:
                    held_positions.update(held_audit["solve_fen_norm"].dropna().astype(str))
                if "original_fen_norm" in held_audit:
                    held_positions.update(held_audit["original_fen_norm"].dropna().astype(str))
            split_row["all_position_fen_overlap_heldout"] = len(split_positions & held_positions)
            split_reports.append(split_row)
            lines.append(f"  FULL {split_name}: illegal={split_row['illegal_lines']}, no-mate={split_row['not_ending_mate']}, mate_n mismatch={split_row['mate_n_mismatch']}, solve-FEN overlap heldout={split_row['solve_fen_overlap_heldout']}")
        else:
            split_reports.append(split_row)
    if split_reports:
        _write_csv(pd.DataFrame(split_reports), "data_split_leakage.csv")
    if n_distribution_rows:
        nd = pd.DataFrame(n_distribution_rows)
        wide = nd.pivot(index="mate_n", columns="split", values="count").fillna(0).astype(int).reset_index()
        _write_csv(wide, "data_n_distribution.csv")
        totals = nd.groupby("split")["count"].sum().to_dict()
        lines.append(f"Split totals: {totals}")
        lines.append(f"Configured SUBSET_SIZE={getattr(config, 'SUBSET_SIZE', 'n/a')}; observed train+val+test={sum(totals.values())}")
    if rating_distribution_rows:
        rd = pd.DataFrame(rating_distribution_rows)
        _write_csv(rd, "data_n_by_rating_bin_distribution.csv")
        # Include count and within-N fraction to see whether train/val/test distributions align.
        rd["n_total"] = rd.groupby(["split", "mate_n"])["count"].transform("sum")
        rd["within_n_fraction"] = rd["count"] / rd["n_total"].replace(0, np.nan)
        _write_csv(rd, "data_n_by_rating_bin_distribution_with_fractions.csv")
        lines.append("Distribuzione mate_n × rating_bin salvata in data_n_by_rating_bin_distribution_with_fractions.csv")

    split_names = list(split_quick_frames)
    for i, left in enumerate(split_names):
        for right in split_names[i+1:]:
            la, rb = split_quick_frames[left], split_quick_frames[right]
            id_overlap = len(set(la["PuzzleId"].astype(str)) & set(rb["PuzzleId"].astype(str))) if "PuzzleId" in la and "PuzzleId" in rb else -1
            fen_overlap = len(set(la["FEN"].astype(str).map(_norm_fen)) & set(rb["FEN"].astype(str).map(_norm_fen))) if "FEN" in la and "FEN" in rb else -1
            lines.append(f"SPLIT OVERLAP {left} × {right}: ID={id_overlap}, original FEN={fen_overlap}")

    if set(source_frames) == {"mixed", "lichess_only"}:
        mixed = source_frames["mixed"].copy()
        lich = source_frames["lichess_only"].copy()
        mix_counts = {(int(n), str(s).lower()): int(c) for (n, s), c in mixed.groupby(["mate_n", "source"]).size().items()}
        expected_mix = {(1,"polgar"):30, (2,"polgar"):30, (3,"polgar"):30, (4,"lichess"):30, (5,"lichess"):30}
        lich_counts = lich.groupby("mate_n").size().to_dict()
        lines.append(f"HELD-OUT mixed composition exact={mix_counts == expected_mix}; counts={mix_counts}")
        lines.append(f"HELD-OUT Lichess-only counts by N={lich_counts}; all source Lichess={bool(lich['source'].astype(str).str.lower().eq('lichess').all()) if 'source' in lich else 'source column absent'}")
        if "PuzzleId" in mixed and "PuzzleId" in lich:
            id_intersection = len(set(mixed.PuzzleId.astype(str)) & set(lich.PuzzleId.astype(str)))
            fens_a = set(mixed.FEN.astype(str).map(_norm_fen))
            fens_b = set(lich.FEN.astype(str).map(_norm_fen))
            lines.append(f"HELD-OUT overlap mixed × Lichess-only: IDs={id_intersection}, original FEN={len(fens_a & fens_b)}")


def _audit_llm_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    records = []
    for row in df.to_dict(orient="records"):
        fen = str(row.get("fen_after_opening", "") or "")
        target = _tokens(row.get("target_moves"))
        proposed = _tokens(row.get("proposed_moves"))
        explanation = str(row.get("llm_explanation", "") or "")
        alternating = _replay(fen, proposed) if fen and proposed else _replay(fen, []) if fen else {"all_moves_legal":False,"ends_in_checkmate":False,"status":"missing_fen"}

        # Alternative convention: proposal lists only the solver's moves and we interleave
        # the reference opponent replies. This is a legality diagnostic only, not proof
        # that the proposed line forces mate against best defence.
        board = None
        solver_legal = bool(fen and proposed)
        solver_first_illegal = None
        solver_illegal_token = ""
        solver_error = ""
        solver_plies = 0
        solver_mate = False
        try:
            board = chess.Board(fen)
            for i, token in enumerate(proposed):
                try:
                    move = chess.Move.from_uci(token)
                except Exception:
                    solver_legal = False; solver_first_illegal = i + 1; solver_illegal_token = token; solver_error = "invalid_uci"; break
                if move not in board.legal_moves:
                    solver_legal = False; solver_first_illegal = i + 1; solver_illegal_token = token; solver_error = "illegal_solver_move"; break
                board.push(move); solver_plies += 1
                if board.is_checkmate():
                    if i < len(proposed) - 1:
                        solver_legal = False; solver_first_illegal = i + 2; solver_illegal_token = proposed[i+1]; solver_error = "moves_after_mate"
                    break
                # target alternates solver, opponent, solver, ...; use its reference opponent reply.
                reply_idx = 2 * i + 1
                if reply_idx < len(target):
                    try:
                        reply = chess.Move.from_uci(target[reply_idx])
                    except Exception:
                        solver_legal = False; solver_first_illegal = i + 1; solver_illegal_token = target[reply_idx]; solver_error = "invalid_reference_reply"; break
                    if reply not in board.legal_moves:
                        solver_legal = False; solver_first_illegal = i + 1; solver_illegal_token = target[reply_idx]; solver_error = "reference_reply_illegal_after_proposal"; break
                    board.push(reply)
            solver_mate = bool(board and board.is_checkmate())
        except Exception as exc:
            solver_legal = False; solver_error = f"board_error:{type(exc).__name__}"

        target_replay = _replay(fen, target) if fen and target else {"all_moves_legal": False, "ends_in_checkmate": False, "status": "empty_or_missing_target"}
        target_solver = target[0::2]
        response_has_solution = bool(re.search(r"\bSOLUTION\s*:", explanation, re.IGNORECASE))
        proposal_empty = len(proposed) == 0
        exact_full = proposed == target
        exact_solver = proposed == target_solver
        rec = {
            "dataset": split_name,
            "PuzzleId": str(row.get("PuzzleId", "")),
            "mate_n": row.get("mate_n"),
            "source": row.get("source", ""),
            "fen_after_opening": fen,
            "target_moves_uci": " ".join(target),
            "proposed_moves_uci": " ".join(proposed),
            "response_text": explanation,
            "target_plies": len(target),
            "target_solver_moves": len(target_solver),
            "proposed_plies": len(proposed),
            "empty_proposal": proposal_empty,
            "solution_tag_present_in_text": response_has_solution,
            "target_legal": target_replay.get("all_moves_legal", False),
            "target_ends_in_checkmate": target_replay.get("ends_in_checkmate", False),
            "target_line_status": target_replay.get("status", ""),
            "proposal_legal_as_alternating_line": alternating.get("all_moves_legal", False),
            "proposal_ends_in_checkmate_as_alternating_line": alternating.get("ends_in_checkmate", False),
            "proposal_alternating_status": alternating.get("status", ""),
            "alternating_first_illegal_ply": alternating.get("first_illegal_ply"),
            "proposal_legal_as_solver_only_with_reference_replies": solver_legal,
            "proposal_ends_in_mate_with_reference_replies": solver_mate,
            "solver_only_first_illegal_ply": solver_first_illegal,
            "solver_only_error": solver_error,
            "exact_full_target": exact_full,
            "exact_solver_moves": exact_solver,
            "exact_either_format": exact_full or exact_solver,
            "shorter_than_full_target": len(proposed) < len(target),
            "shorter_than_required_solver_moves": len(proposed) < len(target_solver),
            "legacy_proposed_legal": _bool(row.get("proposed_legal")),
            "legacy_full_sequence_correct": _bool(row.get("full_sequence_correct")),
            "legacy_first_move_correct": _bool(row.get("first_move_correct")),
            "first_move_matches_target": bool(proposed and target and proposed[0] == target[0]),
            "interpretation": row.get("interpretation", ""),
            "response_time_sec": row.get("response_time_sec", np.nan),
        }
        records.append(rec)
    return pd.DataFrame(records)


def _llm_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, g in audit.groupby("mate_n", dropna=False):
        def mean(col):
            return float(g[col].fillna(False).astype(bool).mean()) if len(g) and col in g else float("nan")
        rows.append({
            "mate_n": n, "n": len(g),
            "first_move_matches_target": mean("first_move_matches_target"),
            "exact_full_target": mean("exact_full_target"),
            "exact_solver_moves": mean("exact_solver_moves"),
            "empty_proposal_count": int(g["empty_proposal"].sum()),
            "target_line_invalid_count": int((~g["target_legal"].astype(bool)).sum()),
            "target_not_mate_count": int((~g["target_ends_in_checkmate"].astype(bool)).sum()),
            "proposal_legal_alternating_count": int(g["proposal_legal_as_alternating_line"].sum()),
            "proposal_ends_mate_alternating_count": int(g["proposal_ends_in_checkmate_as_alternating_line"].sum()),
            "proposal_legal_solver_only_ref_replies_count": int(g["proposal_legal_as_solver_only_with_reference_replies"].sum()),
            "proposal_mates_vs_reference_replies_count": int(g["proposal_ends_in_mate_with_reference_replies"].sum()),
            "shorter_than_full_target_count": int(g["shorter_than_full_target"].sum()),
            "solution_tag_missing_count": int((~g["solution_tag_present_in_text"]).sum()),
            "legacy_proposed_legal_count": int(g["legacy_proposed_legal"].sum()),
        })
    return pd.DataFrame(rows)


def section_llm(lines: list[str]) -> None:
    print("\n[2/5] AUDIT CSV LLM", flush=True)
    summaries = []
    for set_name, files in LLM_FILES.items():
        raw = _load_csv(files["raw"])
        sample = _load_csv(files["sample"])
        if raw is None:
            lines.append(f"LLM {set_name}: raw CSV mancante")
            print(f"  [SKIP] LLM {set_name}: raw CSV mancante")
            continue
        audit = _audit_llm_df(raw, set_name)
        _write_csv(audit, f"llm_row_audit_{set_name}.csv")
        summary = _llm_summary(audit)
        summary.insert(0, "dataset", set_name)
        summaries.append(summary)
        lines.append(f"LLM {set_name}: righe raw={len(raw)}, id unici={raw['PuzzleId'].nunique() if 'PuzzleId' in raw else 'n/a'}")
        lines.append(f"  proposta vuota={int(audit.empty_proposal.sum())}; tag SOLUTION mancante={int((~audit.solution_tag_present_in_text).sum())}; target non legale={int((~audit.target_legal).sum())}; target non termina in matto={int((~audit.target_ends_in_checkmate).sum())}")
        lines.append(f"  proposta legale come linea alternata={int(audit.proposal_legal_as_alternating_line.sum())}/{len(audit)}; linea alternata termina in matto={int(audit.proposal_ends_in_checkmate_as_alternating_line.sum())}/{len(audit)}")
        lines.append(f"  proposta legale come sole mosse solver + risposte target={int(audit.proposal_legal_as_solver_only_with_reference_replies.sum())}/{len(audit)}; termina in matto su risposte target={int(audit.proposal_ends_in_mate_with_reference_replies.sum())}/{len(audit)}")
        lines.append(f"  più corta del target={int(audit.shorter_than_full_target.sum())}/{len(audit)}; esatta a linea completa={int(audit.exact_full_target.sum())}/{len(audit)}; esatta alle sole mosse solver={int(audit.exact_solver_moves.sum())}/{len(audit)}")
        if sample is not None and "PuzzleId" in sample and "PuzzleId" in raw:
            sample_ids = set(sample["PuzzleId"].astype(str))
            raw_ids = set(raw["PuzzleId"].astype(str))
            missing_ids = sorted(sample_ids - raw_ids)
            extra_ids = sorted(raw_ids - sample_ids)
            lines.append(f"  campione canonico={len(sample)}; ID campione mancanti nel raw={len(missing_ids)}; ID extra nel raw={len(extra_ids)}")
            _write_csv(pd.DataFrame({"PuzzleId": missing_ids}), f"llm_missing_from_raw_{set_name}.csv")
        else:
            lines.append("  campione canonico assente o non confrontabile: impossibile identificare richieste mancanti")
        for _, r in summary.iterrows():
            lines.append(f"  N{r['mate_n']}: n={int(r['n'])}; empty={int(r['empty_proposal_count'])}; exact_full={int(r['exact_full_target']*r['n']) if pd.notna(r['exact_full_target']) else 0}; alt-line legal={int(r['proposal_legal_alternating_count'])}; ends-mate={int(r['proposal_ends_mate_alternating_count'])}")
    if summaries:
        _write_csv(pd.concat(summaries, ignore_index=True), "llm_summary_by_n.csv")


def _mcnemar_p(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        p = 1.0
    elif binomtest is not None:
        p = float(binomtest(min(only_a, only_b), n, 0.5, alternative="two-sided").pvalue)
    else:
        p = float("nan")
    return only_a, only_b, p


def _bootstrap_ci_paired(diff: np.ndarray, seed: int = 20261025, reps: int = 4000) -> tuple[float, float]:
    if len(diff) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    n = len(diff)
    for i in range(reps):
        vals[i] = rng.choice(diff, size=n, replace=True).mean()
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def section_ablation(lines: list[str]) -> None:
    print("\n[3/5] ABLATION TIMED vs NO-TIMING", flush=True)
    a = _load_csv(GNN_FILES["timed"])
    b = _load_csv(GNN_FILES["no_timing"])
    if a is None or b is None:
        lines.append("ABLATION: manca uno dei due CSV test_sequence timed/no_timing")
        return
    if a["PuzzleId"].astype(str).duplicated().any() or b["PuzzleId"].astype(str).duplicated().any():
        lines.append("ABLATION ERROR: PuzzleId duplicati nei CSV; confronto appaiato sospeso")
        return
    m = a.merge(b, on="PuzzleId", how="outer", suffixes=("_timed", "_no"), indicator=True)
    if not m["_merge"].eq("both").all():
        lines.append(f"ABLATION ERROR: insiemi PuzzleId diversi; left_only={(m._merge=='left_only').sum()}, right_only={(m._merge=='right_only').sum()}")
        _write_csv(m[m["_merge"] != "both"], "ablation_unmatched_puzzle_ids.csv")
        return
    if "mate_n_timed" in m and "mate_n_no" in m:
        mismatch = int((m["mate_n_timed"] != m["mate_n_no"]).sum())
        lines.append(f"ABLATION: n={len(m)}, ID uguali; differenze mate_n tra csv={mismatch}")
    else:
        mismatch = 0
    rows_overall = []
    rows_n = []
    for metric in ["first_move_correct", "full_sequence_correct"]:
        ca, cb = metric + "_timed", metric + "_no"
        if ca not in m or cb not in m:
            continue
        aa = m[ca].map(_bool).to_numpy(dtype=bool)
        bb = m[cb].map(_bool).to_numpy(dtype=bool)
        only_a, only_b, p = _mcnemar_p(aa, bb)
        ci_lo, ci_hi = _bootstrap_ci_paired(aa.astype(float) - bb.astype(float), seed=20261025 + len(metric))
        rows_overall.append({"metric": metric, "n_pairs": len(m), "timed_correct": int(aa.sum()), "no_timing_correct": int(bb.sum()), "timed_accuracy": float(aa.mean()), "no_timing_accuracy": float(bb.mean()), "delta_pp": float((aa.mean()-bb.mean())*100), "delta_ci95_low_pp": ci_lo*100, "delta_ci95_high_pp": ci_hi*100, "timed_only_correct": only_a, "no_timing_only_correct": only_b, "mcnemar_exact_p": p})
        lines.append(f"ABLATION {metric}: Timed={aa.mean():.4%} ({aa.sum()}/{len(m)}), no-timing={bb.mean():.4%} ({bb.sum()}/{len(m)}), delta={(aa.mean()-bb.mean())*100:+.3f} pp, paired bootstrap 95% CI=[{ci_lo*100:+.3f},{ci_hi*100:+.3f}] pp, McNemar exact p={p:.6g}, discordanti only-timed={only_a}, only-no={only_b}")
        n_col = "mate_n_timed" if "mate_n_timed" in m else "mate_n"
        if n_col in m:
            for n, inds in m.groupby(n_col).groups.items():
                idx = np.asarray(list(inds), dtype=int)
                an, bn = aa[idx], bb[idx]
                x, y, pn = _mcnemar_p(an, bn)
                lo, hi = _bootstrap_ci_paired(an.astype(float)-bn.astype(float), seed=20261025+int(n)+len(metric), reps=2500)
                rows_n.append({"metric": metric, "mate_n": n, "n_pairs": len(idx), "timed_accuracy": float(an.mean()), "no_timing_accuracy": float(bn.mean()), "delta_pp": float((an.mean()-bn.mean())*100), "delta_ci95_low_pp": lo*100, "delta_ci95_high_pp": hi*100, "timed_only_correct": x, "no_timing_only_correct": y, "mcnemar_exact_p_unadjusted": pn})
    if rows_overall:
        _write_csv(pd.DataFrame(rows_overall), "ablation_paired_overall.csv")
    if rows_n:
        ndf = pd.DataFrame(rows_n)
        # Holm adjustment is per metric family, here separately for the five N-level tests.
        ndf["mcnemar_p_holm_by_metric"] = np.nan
        for metric, idxs in ndf.groupby("metric").groups.items():
            idxs = list(idxs)
            pcol = next(
                (c for c in (
                    "mcnemar_p_exact_unadjusted",
                    "mcnemar_exact_p",
                    "mcnemar_p_exact",
                ) if c in ndf.columns),
                None,
            )
            if pcol is None:
                lines.append(
                    f"AVVISO: colonna p-value McNemar assente per {metric}; "
                    "correzione Holm non calcolata."
                )
                continue
            ps = ndf.loc[idxs, pcol].to_numpy(dtype=float)
            order = np.argsort(ps)
            adj = np.empty(len(ps), dtype=float)
            running = 0.0
            for rank, ix in enumerate(order):
                val = min(1.0, (len(ps)-rank)*ps[ix])
                running = max(running, val)
                adj[ix] = running
            ndf.loc[idxs, "mcnemar_p_holm_by_metric"] = adj
        _write_csv(ndf, "ablation_paired_by_n.csv")


def section_gnn_moves(lines: list[str]) -> None:
    print("\n[4/5] ERRORI GNN PER MOSSA", flush=True)
    test = _load_parquet(Path(config.TEST_PARQUET))
    if test is None or not {"PuzzleId", "FEN", "Moves", "mate_n"}.issubset(test.columns):
        lines.append("GNN move-wise: impossibile: puzzles_test.parquet mancante o schema incompleto")
        return
    if "has_opening_move" not in test.columns:
        test["has_opening_move"] = True
    if "source" not in test.columns:
        test["source"] = "lichess"
    test_idx = {str(r["PuzzleId"]): r for r in test.to_dict(orient="records")}
    first_error_rows = []
    step_rows = []
    for variant, path in GNN_FILES.items():
        pred_df = _load_csv(path)
        if pred_df is None:
            lines.append(f"GNN move-wise {variant}: CSV predizioni mancante")
            continue
        missing_ref = 0
        for pr in pred_df.to_dict(orient="records"):
            pid = str(pr.get("PuzzleId", ""))
            ref = test_idx.get(pid)
            if ref is None:
                missing_ref += 1
                continue
            fen = str(ref.get("FEN", ""))
            tokens = _tokens(ref.get("Moves"))
            opening = _has_opening(ref.get("has_opening_move"), ref.get("source"))
            solver_targets = tokens[1::2] if opening else tokens[0::2]
            predicted = _tokens(pr.get("predicted_moves"))
            first_err = None
            for i, move in enumerate(predicted):
                if i >= len(solver_targets) or move != solver_targets[i]:
                    first_err = i + 1
                    break
            if first_err is None and len(predicted) < len(solver_targets):
                first_err = len(predicted) + 1
            all_exact = predicted == solver_targets
            # Replay gold line and predicted line (with target replies) to detect legality/mate status.
            gold_check = _replay(fen, tokens)
            board = None; pred_solver_legal = True; pred_solver_error = ""; pred_solver_mate = False; pred_attempts = 0
            try:
                board = chess.Board(fen)
                if opening and tokens:
                    opening_move = chess.Move.from_uci(tokens[0])
                    if opening_move not in board.legal_moves:
                        raise ValueError("illegal_opening_move")
                    board.push(opening_move)
                opponent_replies = (tokens[2::2] if opening else tokens[1::2])
                for i, token in enumerate(predicted):
                    try:
                        move = chess.Move.from_uci(token)
                    except Exception:
                        pred_solver_legal=False; pred_solver_error="invalid_uci"; break
                    if move not in board.legal_moves:
                        pred_solver_legal=False; pred_solver_error="illegal_predicted_move"; break
                    board.push(move); pred_attempts += 1
                    if board.is_checkmate():
                        if i < len(predicted)-1:
                            pred_solver_legal=False; pred_solver_error="moves_after_mate"
                        break
                    if i < len(opponent_replies):
                        reply = chess.Move.from_uci(opponent_replies[i])
                        if reply not in board.legal_moves:
                            pred_solver_legal=False; pred_solver_error="reference_reply_illegal_after_prediction"; break
                        board.push(reply)
                pred_solver_mate = bool(board and board.is_checkmate())
            except Exception as exc:
                pred_solver_legal = False; pred_solver_error = f"board_error:{type(exc).__name__}"
            first_error_rows.append({
                "variant": variant, "PuzzleId": pid, "mate_n": ref.get("mate_n"),
                "target_solver_moves": len(solver_targets), "predicted_solver_moves": len(predicted),
                "target_length_matches_mate_n": len(solver_targets) == int(ref.get("mate_n")),
                "reference_FEN": fen,
                "reference_Moves_UCI": " ".join(tokens),
                "target_solver_moves_UCI": " ".join(solver_targets),
                "predicted_moves_UCI": " ".join(predicted),
                "first_error_solver_move_index_1based": first_err,
                "first_move_correct_recomputed": bool(predicted and solver_targets and predicted[0] == solver_targets[0]),
                "full_sequence_exact_recomputed": all_exact,
                "full_sequence_flag_from_csv": _bool(pr.get("full_sequence_correct")),
                "gold_line_legal": gold_check.get("all_moves_legal", False),
                "gold_line_ends_in_mate": gold_check.get("ends_in_checkmate", False),
                "predicted_line_legal_with_reference_replies": pred_solver_legal,
                "predicted_line_ends_in_mate_with_reference_replies": pred_solver_mate,
                "prediction_replay_error": pred_solver_error,
                "prediction_length_gt_mate_n": len(predicted) > int(ref.get("mate_n")),
            })
            for i, pred_move in enumerate(predicted):
                if i >= len(solver_targets):
                    equal = False
                else:
                    equal = pred_move == solver_targets[i]
                step_rows.append({"variant": variant, "PuzzleId": pid, "mate_n": int(ref.get("mate_n")), "solver_move_index_1based": i+1, "prediction_available": True, "prediction_matches_reference": bool(equal)})
        lines.append(f"GNN {variant}: righe CSV={len(pred_df)}; PuzzleId senza riferimento in test parquet={missing_ref}")
    if first_error_rows:
        ferr = pd.DataFrame(first_error_rows)
        _write_csv(ferr, "gnn_first_error_per_puzzle.csv")
        err_summary = ferr.groupby(["variant", "mate_n"]).agg(
            n=("PuzzleId", "count"),
            exact_sequence_recomputed=("full_sequence_exact_recomputed", "mean"),
            first_move_recomputed=("first_move_correct_recomputed", "mean"),
            gold_line_invalid=("gold_line_legal", lambda s: int((~s).sum())),
            gold_line_not_mate=("gold_line_ends_in_mate", lambda s: int((~s).sum())),
            target_length_mismatch=("target_length_matches_mate_n", lambda s: int((~s).sum())),
            predictions_longer_than_mate_n=("prediction_length_gt_mate_n", "sum"),
            predicted_line_legal_on_reference_replies=("predicted_line_legal_with_reference_replies", "mean"),
            predicted_line_ends_mate_on_reference_replies=("predicted_line_ends_in_mate_with_reference_replies", "mean"),
        ).reset_index()
        _write_csv(err_summary, "gnn_error_summary_by_n.csv")
        stepdf = pd.DataFrame(step_rows)
        step_summary = stepdf.groupby(["variant", "mate_n", "solver_move_index_1based"]).agg(
            predictions_available=("prediction_available", "sum"),
            matches=("prediction_matches_reference", "sum"),
            conditional_accuracy=("prediction_matches_reference", "mean"),
        ).reset_index()
        _write_csv(step_summary, "gnn_move_accuracy_by_step.csv")
        first_err_counts = ferr.assign(first_error=ferr["first_error_solver_move_index_1based"].fillna(0).astype(int)).groupby(["variant", "mate_n", "first_error"]).size().rename("count").reset_index()
        _write_csv(first_err_counts, "gnn_first_error_distribution.csv")
        n_bad_length = int((~ferr["target_length_matches_mate_n"]).sum())
        lines.append(f"GNN evaluation: righe con numero mosse di riferimento diverso da mate_n={n_bad_length}/{len(ferr)} (sommando entrambe le varianti; può essere doppio conteggio)")


def section_inventory(lines: list[str]) -> None:
    print("\n[5/5] INVENTARIO / COPERTURA CSV", flush=True)
    rows = []
    for p in sorted(RESULTS.glob("*.csv")):
        try:
            df = pd.read_csv(p)
            rows.append({"file": p.name, "rows": len(df), "columns": len(df.columns), "columns_list": " | ".join(df.columns.astype(str))})
        except Exception as exc:
            rows.append({"file": p.name, "rows": None, "columns": None, "columns_list": f"READ_ERROR {exc}"})
    _write_csv(pd.DataFrame(rows), "csv_inventory.csv")
    lines.append(f"Inventario CSV risultati: {len(rows)} file; salvato in csv_inventory.csv")
    # Human-evaluation templates: warn if no ratings filled in yet.
    for p in sorted(RESULTS.glob("human_eval_blind*.csv")):
        df = _load_csv(p)
        if df is None:
            continue
        rating_cols = [c for c in df.columns if "correttezza" in c.lower() or "concisione" in c.lower() or "migliore" in c.lower()]
        filled = int(df[rating_cols].notna().sum().sum()) if rating_cols else 0
        lines.append(f"Human eval {p.name}: righe={len(df)}, valori rubric compilati={filled}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["all", "data", "llm", "gnn", "ablation", "inventory"], default="all")
    parser.add_argument("--full-dataset", action="store_true", help="riproduce tutte le linee di train/validation/test; può richiedere vari minuti")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "CHESS TIMED GNN — REPORT DIAGNOSTICO",
        f"Run UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"Project root: {ROOT}",
        f"Full dataset replay: {args.full_dataset}",
        "Nota: una linea legale che termina in matto NON dimostra da sola che il matto sia forzato contro ogni difesa.",
        "Nota: il campo legacy proposed_legal indica legalità nel formato interpretato dal vecchio evaluator, non prova che il matto sia risolto.",
        "",
    ]
    start = time.time()
    print("=== DIAGNOSTICA SOLA LETTURA — nessun checkpoint o dataset verrà modificato ===", flush=True)
    if args.section in ("all", "data"):
        section_data(lines, args.full_dataset)
    if args.section in ("all", "llm"):
        section_llm(lines)
    if args.section in ("all", "ablation"):
        section_ablation(lines)
    if args.section in ("all", "gnn"):
        section_gnn_moves(lines)
    if args.section in ("all", "inventory"):
        section_inventory(lines)
    lines.append("")
    lines.append(f"Durata report: {time.time()-start:.1f} secondi")
    summary_path = OUT / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "="*72 + "\n".join(lines) + "\n" + "="*72, flush=True)
    print(f"Report e CSV diagnostici salvati in: {OUT}", flush=True)
    print("Nota: il codice non ha modificato dati, checkpoint o risultati originali.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
