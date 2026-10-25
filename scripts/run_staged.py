"""Pipeline staged per Colab, con checkpoint persistenti fra sessioni.

Uso:
  python -u -m scripts.run_staged --stage timed
  python -u -m scripts.run_staged --stage no-timing
  python -u -m scripts.run_staged --stage compare

Rilanciare lo STESSO comando dopo una disconnessione riprende l'ultima epoca
completata; le fasi concluse non vengono ripetute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import config
from src.data_pipeline import run_phase1, check_no_leakage_from_files, print_n_distribution
from src.train import (
    device,
    train_one_variant,
    run_phase3,
)
from src.llm_baseline import (
    PROMPT_TEMPLATE,
    get_llm_client,
    load_evaluation_set,
    run_llm_evaluation,
    print_llm_metrics,
)
from src.compare import run_phase4_comparison


DATA_FILES = [
    config.TRAIN_PARQUET,
    config.VAL_PARQUET,
    config.TEST_PARQUET,
    config.HELD_OUT_PARQUET,
    config.HELD_OUT_LICHESS_PARQUET,
]


def _atomic_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_state() -> dict:
    if not os.path.exists(config.STAGED_STATE_JSON):
        return {
            "version": 2,
            "config": _config_signature(),
            "stages": {},
        }
    with open(config.STAGED_STATE_JSON, "r", encoding="utf-8") as f:
        state = json.load(f)
    saved_cfg = state.get("config", {})
    current_cfg = _config_signature()
    if saved_cfg and saved_cfg != current_cfg:
        # Migrazione controllata da una versione precedente della pipeline:
        # se tutti i parametri già esistenti coincidono e sono soltanto aggiunti
        # nuovi campi di schema/stratificazione, si azzera lo stato staged. I
        # checkpoint della nuova metodologia hanno nomi diversi e non vengono riusati.
        is_compatible_legacy_state = all(
            key in current_cfg and current_cfg[key] == value
            for key, value in saved_cfg.items()
        )
        if is_compatible_legacy_state:
            print(
                "[staged] Trovato uno stato della pipeline precedente. "
                "Creo un nuovo stato per subset totale 300k bilanciato per mate_n × rating; "
                "i checkpoint v3 avranno nomi separati e quelli vecchi non saranno riutilizzati.",
                flush=True,
            )
            state = {"version": 2, "config": current_cfg, "stages": {}}
            _atomic_json(config.STAGED_STATE_JSON, state)
            return state
        raise RuntimeError(
            "La configurazione di questa pipeline non coincide con quella salvata "
            f"su Drive.\nSalvata: {saved_cfg}\nAttuale: {current_cfg}\n"
            "Per mantenere la riproducibilità non continuo con configurazioni miste. "
            "Ripristina la configurazione precedente oppure avvia un nuovo esperimento "
            "in una cartella progetto separata."
        )
    state.setdefault("stages", {})
    return state


def _save_state(stage: str, status: str, details: dict | None = None) -> None:
    state = _load_state()
    record = {
        "status": status,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if details:
        record.update(details)
    state["stages"][stage] = record
    state["config"] = _config_signature()
    _atomic_json(config.STAGED_STATE_JSON, state)


def _config_signature() -> dict:
    return {
        "subset_size": config.SUBSET_SIZE,
        "data_pipeline_schema_version": config.DATA_PIPELINE_SCHEMA_VERSION,
        "subset_rating_bins": config.SUBSET_RATING_BINS,
        "split_test_size": config.SPLIT_TEST_SIZE,
        "split_val_of_temp": config.SPLIT_VAL_OF_TEMP,
        "num_epochs": config.NUM_EPOCHS,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": config.EARLY_STOPPING_MIN_DELTA,
        "hidden_dim": config.HIDDEN_DIM,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "graph_cache_max_items": config.GRAPH_CACHE_MAX_ITEMS,
        "random_seed": config.RANDOM_SEED,
        "full_seq_eval_size": config.FULL_SEQ_EVAL_SIZE,
    }


def _marker_path(ckpt_path: str) -> str:
    return ckpt_path + ".complete.json"


def _read_valid_marker(ckpt_path: str) -> bool:
    marker = _marker_path(ckpt_path)
    resume = ckpt_path + ".resume.pt"
    if not (os.path.exists(marker) and os.path.exists(ckpt_path)):
        return False
    try:
        with open(marker, "r", encoding="utf-8") as f:
            payload = json.load(f)
        expected_timing = os.path.normpath(ckpt_path) == os.path.normpath(config.TIMED_CKPT)
        expected = {
            "complete": True,
            "include_timing": expected_timing,
            "subset_size": config.SUBSET_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": config.EARLY_STOPPING_MIN_DELTA,
            "learning_rate": config.LEARNING_RATE,
            "random_seed": config.RANDOM_SEED,
            "graph_cache_max_items": config.GRAPH_CACHE_MAX_ITEMS,
            "hidden_dim": config.HIDDEN_DIM,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "batch_size": config.BATCH_SIZE,
            "data_pipeline_schema_version": config.DATA_PIPELINE_SCHEMA_VERSION,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            print(f"[checkpoint] Marker non compatibile con la configurazione corrente: {marker}")
            os.remove(marker)
            return False
    except Exception as exc:
        print(f"[checkpoint] Marker non leggibile ({exc}); non lo considero completo.")
        try:
            os.remove(marker)
        except OSError:
            pass
        return False
    # Il marker viene scritto solo dopo la fine del training. Se Colab si è
    # interrotto prima di cancellare il resume, è sicuro completare la pulizia.
    if os.path.exists(resume):
        os.remove(resume)
    return True


def _stage_checkpoint(stage: str) -> str:
    return config.TIMED_CKPT if stage == "timed" else config.NO_TIMING_CKPT


def _datasets_are_readable() -> bool:
    required_columns = {
        config.TRAIN_PARQUET: {"PuzzleId", "FEN", "Moves", "mate_n", "rating_norm", "rating_bin"},
        config.VAL_PARQUET: {"PuzzleId", "FEN", "Moves", "mate_n", "rating_norm", "rating_bin"},
        config.TEST_PARQUET: {"PuzzleId", "FEN", "Moves", "mate_n", "rating_norm", "rating_bin"},
        config.HELD_OUT_PARQUET: {"PuzzleId", "FEN", "Moves", "mate_n", "rating_norm", "source"},
        config.HELD_OUT_LICHESS_PARQUET: {"PuzzleId", "FEN", "Moves", "mate_n", "rating_norm", "source"},
    }
    try:
        for path, required in required_columns.items():
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                return False
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows <= 0 or not required.issubset(set(parquet.schema.names)):
                return False

        # Verifica che la composizione del misto sia quella storica e che il
        # secondo held-out sia soltanto Lichess con 30 puzzle per ciascun N.
        df_m = pd.read_parquet(config.HELD_OUT_PARQUET,
                               columns=["PuzzleId", "mate_n", "source"])
        mixed_expected = {
            (1, "polgar"): config.HELD_OUT_PER_N,
            (2, "polgar"): config.HELD_OUT_PER_N,
            (3, "polgar"): config.HELD_OUT_PER_N,
            (4, "lichess"): config.HELD_OUT_PER_N,
            (5, "lichess"): config.HELD_OUT_PER_N,
        }
        mixed_counts = {
            (int(n), str(source).lower()): int(count)
            for (n, source), count in df_m.groupby(["mate_n", "source"]).size().items()
        }
        if mixed_counts != mixed_expected:
            return False

        df_l = pd.read_parquet(config.HELD_OUT_LICHESS_PARQUET,
                               columns=["PuzzleId", "mate_n", "source"])
        if not (df_l["source"].astype(str).str.lower() == "lichess").all():
            return False
        counts = {int(n): int(count) for n, count in df_l.groupby("mate_n").size().items()}
        expected_counts = {int(n): config.HELD_OUT_PER_N for n in config.HELD_OUT_LICHESS_ONLY_N}
        if counts != expected_counts:
            return False
        if set(df_m["PuzzleId"].astype(str)) & set(df_l["PuzzleId"].astype(str)):
            return False

        if not os.path.isfile(config.DATA_PIPELINE_MANIFEST_JSON):
            return False
        with open(config.DATA_PIPELINE_MANIFEST_JSON, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("schema_version") != config.DATA_PIPELINE_SCHEMA_VERSION:
            return False
        if manifest.get("counts", {}).get("held_out_lichess_only") != len(df_l):
            return False
        counts = manifest.get("counts", {})
        subset_design = manifest.get("subset_design", {})
        actual_subset = int(counts.get("subset_total", -1))
        if actual_subset <= 0 or actual_subset != sum(
            int(counts.get(k, 0)) for k in ("train", "validation", "test")
        ):
            return False
        if int(subset_design.get("target_total", -1)) != config.SUBSET_SIZE:
            return False
        if int(subset_design.get("rating_bins_global_quantiles", -1)) != config.SUBSET_RATING_BINS:
            return False
        # Per 300k sono attesi esattamente 240k/30k/30k. Se il pool fosse
        # più piccolo, il manifest registra comunque il totale effettivo.
        if actual_subset == config.SUBSET_SIZE:
            if (int(counts.get("train", 0)), int(counts.get("validation", 0)),
                    int(counts.get("test", 0))) != (
                    int(round(config.SUBSET_SIZE * (1 - config.SPLIT_TEST_SIZE))),
                    int(round(config.SUBSET_SIZE * config.SPLIT_TEST_SIZE * (1 - config.SPLIT_VAL_OF_TEMP))),
                    int(round(config.SUBSET_SIZE * config.SPLIT_TEST_SIZE * config.SPLIT_VAL_OF_TEMP)),
            ):
                return False
        return True
    except Exception:
        return False


def _ensure_datasets() -> None:
    if not _datasets_are_readable():
        print("[dati] Dataset assenti/incompleti: eseguo la pipeline dati.", flush=True)
        run_phase1()
    else:
        print("[dati] Train/validation/test ed entrambi gli held-out già presenti e verificati: salto la preparazione.", flush=True)
    if not _datasets_are_readable():
        raise FileNotFoundError(
            "La pipeline dati non ha creato Parquet leggibili e coerenti. "
            f"Controlla i file in {config.DATA_DIR} e rilancia lo stage timed."
        )
    check_no_leakage_from_files()
    _save_state("data", "complete", {"files": [os.path.basename(p) for p in DATA_FILES]})


def _load_training_frames():
    df_train = pd.read_parquet(config.TRAIN_PARQUET)
    df_val = pd.read_parquet(config.VAL_PARQUET)
    df_test = pd.read_parquet(config.TEST_PARQUET)
    print(
        f"[dati] Train={len(df_train):,} | val={len(df_val):,} | "
        f"test={len(df_test):,} puzzle | device={device}",
        flush=True,
    )
    print_n_distribution(
        {"train": df_train, "val": df_val, "test": df_test},
        "Distribuzione puzzle disponibile",
    )
    actual_subset = len(df_train) + len(df_val) + len(df_test)
    expected_train = int(round(actual_subset * (1 - config.SPLIT_TEST_SIZE)))
    print(
        f"[dati] Subset totale pre-split={actual_subset:,}/{config.SUBSET_SIZE:,}; "
        f"split effettivo train={len(df_train):,}, validation={len(df_val):,}, "
        f"test={len(df_test):,}. La GNN userà tutti i {len(df_train):,} puzzle train "
        f"e tutti i {len(df_val):,} puzzle validation (nessun sottocampionamento aggiuntivo).",
        flush=True,
    )
    if actual_subset >= config.SUBSET_SIZE and len(df_train) != expected_train:
        raise RuntimeError(
            "Split non coerente con SUBSET_SIZE totale. Rilancia la pipeline dati; "
            "il training non parte per evitare una distribuzione errata."
        )
    return df_train, df_val


def _invalidate_comparison_outputs() -> None:
    """Rimuove solo marker/cache delle valutazioni che dipendono dai checkpoint GNN."""
    paths = [
        config.STAGED_ABLATION_DONE_JSON,
        config.STAGED_COMPARISON_DONE_JSON,
        config.STAGED_COMPARISON_MIXED_DONE_JSON,
        config.STAGED_COMPARISON_LICHESS_DONE_JSON,
        config.STAGED_SEQ_TIMED_CSV,
        config.STAGED_SEQ_NO_TIMING_CSV,
        config.ABLATION_CSV,
        config.PAIRED_RESULTS_CSV,
        config.PAIRED_RESULTS_LICHESS_CSV,
    ]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
    state = _load_state()
    state.get("stages", {}).pop("compare", None)
    _atomic_json(config.STAGED_STATE_JSON, state)


def _run_training_stage(stage: str, reset: bool = False) -> None:
    if stage not in ("timed", "no-timing"):
        raise ValueError(stage)
    ckpt_path = _stage_checkpoint(stage)
    resume_path = ckpt_path + ".resume.pt"
    marker_path = _marker_path(ckpt_path)

    if reset:
        for path in (ckpt_path, resume_path, marker_path, resume_path + ".tmp", ckpt_path + ".tmp", marker_path + ".tmp"):
            if os.path.exists(path):
                os.remove(path)
        _invalidate_comparison_outputs()
        print(f"[{stage}] Checkpoint della fase {stage} rimossi; risultati di confronto invalidati.")

    # Il marker viene scritto soltanto dopo la conclusione del training. Se la
    # sessione è terminata nell'intervallo fra marker e pulizia del resume,
    # completa la pulizia invece di proseguire un modello che aveva già early-stoppato.
    if os.path.exists(marker_path) and os.path.exists(ckpt_path) and os.path.exists(resume_path):
        os.remove(resume_path)

    if stage == "no-timing" and not _read_valid_marker(config.TIMED_CKPT):
        raise RuntimeError(
            "La fase timed non risulta completata. Prima esegui `--stage timed` e "
            "attendi il messaggio di completamento; poi passa a `--stage no-timing`."
        )

    if _read_valid_marker(ckpt_path):
        print(f"[{stage}] Fase già completata: checkpoint e marker validi trovati. La salto.")
        _save_state(stage, "complete", {"checkpoint": os.path.basename(ckpt_path)})
        return

    # Un checkpoint BEST senza resume né marker non prova che il training sia
    # finito. Con i nomi staged è probabilmente un residuo incompleto: non usarlo.
    if os.path.exists(ckpt_path) and not os.path.exists(resume_path) and not os.path.exists(marker_path):
        print(f"[{stage}] Trovato un checkpoint senza marker di completamento; lo considero incompleto e lo ricreo.")
        os.remove(ckpt_path)

    _save_state(stage, "in_progress", {"checkpoint": os.path.basename(ckpt_path)})
    _ensure_datasets()
    check_no_leakage_from_files()
    df_train_run, df_val_run = _load_training_frames()

    print(
        f"\n[{stage}] CONFIG: subset={config.SUBSET_SIZE:,} | epoche max={config.NUM_EPOCHS} | "
        f"early stopping={config.EARLY_STOPPING_PATIENCE} | batch={config.BATCH_SIZE}",
        flush=True,
    )
    train_one_variant(
        df_train_run,
        df_val_run,
        include_timing=(stage == "timed"),
        ckpt_path=ckpt_path,
        tag=stage,
    )
    if not _read_valid_marker(ckpt_path):
        raise RuntimeError(
            f"[{stage}] Il training è ritornato ma non trovo checkpoint/marker coerenti. "
            "Non segno la fase come completata."
        )
    _save_state(stage, "complete", {"checkpoint": os.path.basename(ckpt_path)})
    print(f"\n[{stage}] COMPLETATO. File conservati in {config.CKPT_DIR}. La prossima fase NON è stata avviata.", flush=True)


def _stage_done(stage: str) -> bool:
    ckpt_path = _stage_checkpoint(stage)
    return _read_valid_marker(ckpt_path)


def _current_llm_signature() -> dict:
    prompt_hash = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    return {
        "backend": config.LLM_BACKEND,
        "model_name": config.LLM_MODEL_NAME,
        "n_samples": config.LLM_N_SAMPLES,
        "max_tokens": config.LLM_MAX_TOKENS,
        "prompt_sha256": prompt_hash,
    }


def _ensure_llm_signature() -> None:
    """Evita di mescolare nel CSV risposte generate da modelli/prompt diversi."""
    path = config.STAGED_LLM_SIGNATURE_JSON
    current = _current_llm_signature()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                previous = json.load(f)
        except Exception:
            previous = None
        if config.LLM_FRESH_START and os.path.exists(config.STAGED_COMPARISON_DONE_JSON):
            os.remove(config.STAGED_COMPARISON_DONE_JSON)
        if previous != current:
            if not config.LLM_FRESH_START:
                raise RuntimeError(
                    "La configurazione LLM (backend/modello/prompt/campione) è diversa "
                    "da quella della run precedente. Per non mescolare risposte di modelli "
                    "diversi, rilancia una sola volta con `LLM_FRESH_START=1` per iniziare "
                    "una nuova valutazione; il CSV precedente verrà conservato come backup."
                )
            _atomic_json(path, current)
            if os.path.exists(config.STAGED_COMPARISON_DONE_JSON):
                os.remove(config.STAGED_COMPARISON_DONE_JSON)
            print("[compare] Nuova firma LLM registrata; la valutazione precedente sarà archiviata.")
    else:
        if config.LLM_FRESH_START and os.path.exists(config.STAGED_COMPARISON_DONE_JSON):
            os.remove(config.STAGED_COMPARISON_DONE_JSON)
        if os.path.exists(config.LLM_RAW_CSV) and not config.LLM_FRESH_START:
            print("[compare] Nota: riuso il CSV LLM preesistente; non esiste una firma storica "
                  "per verificare il modello con cui fu prodotto.")
        _atomic_json(path, current)


def _run_one_heldout_comparison(label: str, overrides: dict) -> bool:
    """Esegue o riprende il confronto GNN-LLM per UNO specifico held-out.

    I percorsi vengono temporaneamente commutati: ciò permette di riutilizzare
    le funzioni correnti mantenendo CSV, campioni, firme e marker separati.
    Ritorna False se la baseline LLM è ancora incompleta (es. quota API finita).
    """
    old_values = {key: getattr(config, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(config, key, value)

        done_path = config.STAGED_COMPARISON_DONE_JSON
        paired_path = config.PAIRED_RESULTS_CSV
        if (not config.LLM_FRESH_START and os.path.exists(done_path)
                and os.path.exists(paired_path)):
            print(f"[compare:{label}] Già completato: riuso {os.path.basename(paired_path)}.", flush=True)
            return True

        _ensure_llm_signature()
        print(f"\n[compare:{label}] Held-out: {config.HELD_OUT_PARQUET}", flush=True)
        df_eval = load_evaluation_set(n_samples=config.LLM_N_SAMPLES)
        client = get_llm_client(backend=config.LLM_BACKEND)
        df_llm = run_llm_evaluation(
            df_eval,
            client,
            backend=config.LLM_BACKEND,
            model_name=config.LLM_MODEL_NAME,
        )
        expected_ids = set(df_eval["PuzzleId"].astype(str))
        completed_ids = set(df_llm["PuzzleId"].astype(str)) if "PuzzleId" in df_llm else set()
        missing_ids = expected_ids - completed_ids
        if missing_ids:
            completed = len(expected_ids & completed_ids)
            print(
                f"\n[compare:{label}] Baseline LLM incompleta: {completed}/"
                f"{len(expected_ids)} puzzle. I risultati sono salvati; rilancia "
                "`--stage compare` in una sessione futura per continuare.",
                flush=True,
            )
            _save_state("compare", "in_progress", {
                "substage": f"llm_{label}",
                "llm_completed": completed,
                "llm_expected": len(expected_ids),
            })
            return False

        df_llm = df_llm[df_llm["PuzzleId"].astype(str).isin(expected_ids)].copy()
        print_llm_metrics(df_llm)
        run_phase4_comparison(df_llm, df_eval)
        _atomic_json(done_path, {
            "complete": True,
            "held_out": label,
            "held_out_path": config.HELD_OUT_PARQUET,
            "held_out_puzzles": int(len(df_eval)),
            "config": _config_signature(),
            "llm_backend": config.LLM_BACKEND,
            "llm_model": config.LLM_MODEL_NAME,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"[compare:{label}] COMPLETATO. Risultati in {config.RESULTS_DIR}.", flush=True)
        return True
    finally:
        for key, value in old_values.items():
            setattr(config, key, value)


def _run_compare() -> None:
    if not _stage_done("timed"):
        raise RuntimeError("Manca il checkpoint completo Timed GNN. Esegui prima `--stage timed`.")
    if not _stage_done("no-timing"):
        raise RuntimeError("Manca il checkpoint completo no-timing. Esegui prima `--stage no-timing`.")
    if not os.path.exists(config.HELD_OUT_PARQUET) or not os.path.exists(config.HELD_OUT_LICHESS_PARQUET):
        raise FileNotFoundError(
            "Servono entrambi gli held-out. Rilancia prima `python -m scripts.run_pipeline` "
            "oppure lo stage timed per rigenerare i dataset."
        )

    # Fase A: test interno e ablation timed vs no-timing (una volta sola).
    # Ha checkpoint incrementali propri, quindi si può riprendere da Colab.
    if not (os.path.exists(config.STAGED_ABLATION_DONE_JSON) and os.path.exists(config.ABLATION_CSV)):
        _save_state("compare", "in_progress", {"substage": "ablation_test_interno"})
        print("[compare] Valuto test set interno e ablation timed vs no-timing.", flush=True)
        run_phase3(skip_training=True)
        _atomic_json(config.STAGED_ABLATION_DONE_JSON, {
            "complete": True,
            "config": _config_signature(),
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    else:
        print("[compare] Ablation interna già completata; riuso i risultati salvati.", flush=True)

    # Fase B: confronto sul held-out misto (stesso file/output storici).
    mixed_overrides = {
        "HELD_OUT_PARQUET": config.HELD_OUT_PARQUET,
        "STAGED_COMPARISON_DONE_JSON": config.STAGED_COMPARISON_MIXED_DONE_JSON,
        "STAGED_LLM_SIGNATURE_JSON": config.STAGED_LLM_SIGNATURE_JSON,
        "LLM_RAW_CSV": config.LLM_RAW_CSV,
        "LLM_EVAL_SAMPLE_CSV": config.LLM_EVAL_SAMPLE_CSV,
        "PAIRED_RESULTS_CSV": config.PAIRED_RESULTS_CSV,
        "ACC_BY_N_CSV": config.ACC_BY_N_CSV,
        "BLIND_EVAL_CSV": config.BLIND_EVAL_CSV,
        "BLIND_EVAL_ANSWER_KEY_CSV": config.BLIND_EVAL_ANSWER_KEY_CSV,
        "CONFUSION_MATRICES_PATH": config.CONFUSION_MATRICES_PATH,
        "CONFUSION_MATRICES_CSV": config.CONFUSION_MATRICES_CSV,
        "COMPARISON_PLOT_PATH": config.COMPARISON_PLOT_PATH,
    }
    mixed_ok = _run_one_heldout_comparison("misto", mixed_overrides)
    if not mixed_ok:
        return

    # Fase C: lo stesso confronto su Lichess-only, con ogni artefatto separato.
    lichess_overrides = {
        "HELD_OUT_PARQUET": config.HELD_OUT_LICHESS_PARQUET,
        "STAGED_COMPARISON_DONE_JSON": config.STAGED_COMPARISON_LICHESS_DONE_JSON,
        "STAGED_LLM_SIGNATURE_JSON": config.STAGED_LLM_SIGNATURE_LICHESS_JSON,
        "LLM_RAW_CSV": config.LLM_RAW_LICHESS_CSV,
        "LLM_EVAL_SAMPLE_CSV": config.LLM_EVAL_SAMPLE_LICHESS_CSV,
        "PAIRED_RESULTS_CSV": config.PAIRED_RESULTS_LICHESS_CSV,
        "ACC_BY_N_CSV": config.ACC_BY_N_LICHESS_CSV,
        "BLIND_EVAL_CSV": config.BLIND_EVAL_LICHESS_CSV,
        "BLIND_EVAL_ANSWER_KEY_CSV": config.BLIND_EVAL_ANSWER_KEY_LICHESS_CSV,
        "CONFUSION_MATRICES_PATH": config.CONFUSION_MATRICES_LICHESS_PATH,
        "CONFUSION_MATRICES_CSV": config.CONFUSION_MATRICES_LICHESS_CSV,
        "COMPARISON_PLOT_PATH": config.COMPARISON_LICHESS_PLOT_PATH,
    }
    lichess_ok = _run_one_heldout_comparison("lichess-only", lichess_overrides)
    if not lichess_ok:
        return

    # Marker aggregato: compare è completa soltanto quando ENTRAMBI i test sono conclusi.
    _atomic_json(config.STAGED_COMPARISON_DONE_JSON, {
        "complete": True,
        "held_outs": [
            {"name": "misto", "file": os.path.basename(config.HELD_OUT_PARQUET),
             "results": os.path.basename(config.PAIRED_RESULTS_CSV)},
            {"name": "lichess-only", "file": os.path.basename(config.HELD_OUT_LICHESS_PARQUET),
             "results": os.path.basename(config.PAIRED_RESULTS_LICHESS_CSV)},
        ],
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _save_state("compare", "complete", {
        "paired_results_mixed": os.path.basename(config.PAIRED_RESULTS_CSV),
        "paired_results_lichess": os.path.basename(config.PAIRED_RESULTS_LICHESS_CSV),
        "llm_model": config.LLM_MODEL_NAME,
    })
    print("\n[compare] COMPLETATO: held-out misto e held-out Lichess-only.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline GNN a fasi, riprendibile su Colab/Drive.")
    parser.add_argument("--stage", required=True, choices=("timed", "no-timing", "compare"),
                        help="Fase da eseguire; le fasi precedenti devono essere completate.")
    parser.add_argument("--reset-stage", action="store_true",
                        help="Rimuove checkpoint/resume/marker della fase di training indicata e la riavvia da zero.")
    args = parser.parse_args()

    print(
        f"Pipeline staged | stage={args.stage} | subset={config.SUBSET_SIZE:,} | "
        f"epochs={config.NUM_EPOCHS} | patience={config.EARLY_STOPPING_PATIENCE} | "
        f"device={device}",
        flush=True,
    )
    # Blocca incompatibilità prima di rimuovere checkpoint o modificare risultati.
    _load_state()
    if args.stage in ("timed", "no-timing"):
        _run_training_stage(args.stage, reset=args.reset_stage)
    else:
        if args.reset_stage:
            raise ValueError("--reset-stage è ammesso soltanto per timed/no-timing.")
        _run_compare()


if __name__ == "__main__":
    main()
