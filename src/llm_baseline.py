"""
src/llm_baseline.py
---------------------
Fase 4, parte A — Baseline LLM (equivalente alle sezioni 0-6 di
04_llm_baseline_comparison.ipynb).

Cosa fa:
1. Costruisce il prompt "Given this FEN: [FEN], find mate in n moves"
2. Chiama l'LLM (Anthropic o OpenAI) con retry
3. Fa il parsing della risposta in una sequenza di mosse UCI
4. Verifica la soluzione con python-chess (legalità, prima mossa,
   sequenza completa)

La API key NON va mai scritta nel codice: si legge da variabile
d'ambiente (ANTHROPIC_API_KEY o OPENAI_API_KEY). Vedi README.md.
"""

import os
import re
import time
from getpass import getpass

import chess
import pandas as pd
from tqdm.auto import tqdm

import config
from src.chess_encoding import get_position_after_opening

PROMPT_TEMPLATE = """You are a chess expert. Analyze this position and find the forced checkmate.

FEN: {fen}
Side to move: {side}
The position has a forced checkmate in exactly {n} move(s) for the side to move.

First explain your reasoning step by step, briefly (2-4 sentences maximum —
do not write a long move-by-move derivation).

Then finish your answer with exactly one line in this format:

SOLUTION: <move1> <move2> <move3> ...

Rules for that line, follow them exactly:
- Use UCI notation (from-square + to-square, e.g. e2e4, and the promotion
  piece if any, e.g. e7e8q). Do NOT use algebraic/SAN notation (Qxh7#).
- List the ENTIRE forced line, ALTERNATING your moves and the opponent's
  best replies, and end with the checkmating move.
- A mate in 1 is therefore 1 move on that line; a mate in 2 is 3 moves
  (yours, opponent's, yours); a mate in 3 is 5 moves; and so on.
- Put nothing else on that line: no move numbers, no commentary, no dots.

Example of a correctly formatted line for a mate in 2:
SOLUTION: d8h4 g5h4 c1g5

"""

SOLUTION_LINE_RE = re.compile(r"SOLUTION:\s*(.+)")


def build_prompt(fen: str, n: int) -> str:
    side = "White" if fen.split()[1] == "w" else "Black"
    return PROMPT_TEMPLATE.format(fen=fen, side=side, n=n)


def get_llm_client(backend: str = config.LLM_BACKEND):
    """Crea il client per il backend scelto, leggendo la API key dall'ambiente
    (o chiedendola in modo sicuro con getpass se non presente)."""
    if backend == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY") or getpass("Anthropic API key: ")
        return anthropic.Anthropic(api_key=api_key)
    elif backend == "openai":
        import openai
        api_key = os.environ.get("OPENAI_API_KEY") or getpass("OpenAI API key: ")
        return openai.OpenAI(api_key=api_key)
    elif backend == "gemini":
        # Gratuito tramite Google AI Studio: aistudio.google.com -> "Get API key".
        from google import genai
        api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                   or getpass("Gemini API key: "))
        return genai.Client(api_key=api_key)
    elif backend == "groq":
        # Gratuito tramite console.groq.com -> API Keys -> Create API Key.
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY") or getpass("Groq API key: ")
        return Groq(api_key=api_key)
    else:
        raise ValueError("backend deve essere 'anthropic', 'openai', 'gemini' o 'groq'")


class DailyQuotaExceededError(RuntimeError):
    """Sollevato quando l'errore indica chiaramente una quota GIORNALIERA
    esaurita (non un limite al minuto): in questo caso non ha senso
    ritentare, perché il reset avviene solo il giorno dopo."""
    pass


def _is_daily_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "per day" in msg or "daily" in msg


def call_llm(client, prompt: str, backend: str = config.LLM_BACKEND,
             model_name: str = config.LLM_MODEL_NAME,
             max_retries: int = config.LLM_MAX_RETRIES,
             max_tokens: int = config.LLM_MAX_TOKENS) -> str:
    for attempt in range(max_retries):
        try:
            if backend == "anthropic":
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(block.text for block in resp.content if hasattr(block, "text"))
            elif backend == "gemini":
                from google.genai import types
                gen_config = types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=config.GEMINI_THINKING_BUDGET
                    ),
                )
                try:
                    resp = client.models.generate_content(
                        model=model_name, contents=prompt, config=gen_config,
                    )
                except Exception:
                    # Alcuni modelli/versioni potrebbero non supportare
                    # thinking_config: riprova senza, nello stesso tentativo
                    # (non consuma un retry).
                    resp = client.models.generate_content(
                        model=model_name, contents=prompt,
                        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
                    )
                return resp.text
            elif backend == "groq":
                # reasoning_effort="low": riduce il canale di ragionamento
                # interno nascosto (non e' il testo che chiediamo nel
                # prompt, e' un canale separato che i modelli gpt-oss usano
                # PRIMA di scrivere la risposta visibile). reasoning_format
                #="hidden" evita che quel canale competa ulteriormente con
                # lo spazio della risposta visibile. max_completion_tokens
                # e' il nome corretto del parametro per i modelli
                # "reasoning" lato Groq (max_tokens resta supportato come
                # alias ma e' deprecato per questi modelli).
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        max_completion_tokens=max_tokens,
                        reasoning_effort=config.GROQ_REASONING_EFFORT,
                        reasoning_format="hidden",
                        messages=[{"role": "user", "content": prompt}],
                    )
                except Exception:
                    # Modelli Groq non-reasoning non accettano questi
                    # parametri: riprova nello stile "classico", senza
                    # consumare un retry.
                    resp = client.chat.completions.create(
                        model=model_name,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                return resp.choices[0].message.content
            else:  # openai
                resp = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content
        except Exception as e:
            if _is_daily_quota_error(e):
                # Quota GIORNALIERA esaurita: ritentare non serve a nulla
                # (si resetta solo domani), quindi usciamo subito invece
                # di sprecare `max_retries` tentativi con attese crescenti.
                raise DailyQuotaExceededError(str(e)) from e
            wait = 2 ** attempt
            print(f"  [tentativo {attempt + 1}/{max_retries}] errore: {e} -> retry tra {wait}s")
            time.sleep(wait)
    raise RuntimeError("Chiamata LLM fallita dopo tutti i retry")


def parse_solution(response_text: str, starting_fen: str = None):
    """Estrae la lista di mosse UCI dalla riga 'SOLUTION: ...'.

    1) Prova diretta in notazione UCI (quella richiesta esplicitamente nel
       prompt).
    2) Fallback: alcuni modelli rispondono in notazione SAN (es. 'Qxh4#')
       nonostante la richiesta esplicita di UCI — è un comportamento comune
       nei LLM per gli scacchi, imparato dallo stile prevalente nei loro
       dati di training. Se il fallback UCI non trova nulla e viene passata
       `starting_fen`, si prova a interpretare i token come SAN giocandoli
       uno a uno su una scacchiera (ci si ferma alla prima mossa non
       valida/ambigua).
    """
    match = None
    for line in response_text.splitlines():
        m = SOLUTION_LINE_RE.search(line)
        if m:
            match = m.group(1).strip()
    if match is None:
        return None

    candidate_tokens = re.findall(r"[a-h][1-8][a-h][1-8][qrbn]?", match.lower())
    if candidate_tokens:
        return candidate_tokens

    if starting_fen is None:
        return None
    try:
        board = chess.Board(starting_fen)
    except ValueError:
        return None

    san_tokens = re.findall(r"O-O-O[+#]?|O-O[+#]?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?", match)
    uci_tokens = []
    for tok in san_tokens:
        try:
            mv = board.parse_san(tok)
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError, chess.InvalidMoveError):
            break
        uci_tokens.append(mv.uci())
        board.push(mv)
    return uci_tokens if uci_tokens else None


def _play_line(fen: str, uci_list) -> bool:
    """True se tutte le mosse della lista sono legali una dopo l'altra a
    partire da `fen`."""
    board = chess.Board(fen)
    for uci in uci_list:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return False
        if move not in board.legal_moves:
            return False
        board.push(move)
    return True


def _play_solver_only(fen: str, proposed, target_moves) -> bool:
    """True se `proposed` e' legale interpretandola come SOLE mosse del
    solver, intercalando le risposte forzate dell'avversario prese dalla
    soluzione di riferimento."""
    board = chess.Board(fen)
    opponent_replies = list(target_moves[1::2])
    for i, uci in enumerate(proposed):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return False
        if move not in board.legal_moves:
            return False
        board.push(move)
        if i < len(opponent_replies):
            reply = chess.Move.from_uci(opponent_replies[i])
            if reply not in board.legal_moves:
                return False
            board.push(reply)
    return True


def evaluate_llm_solution(fen_after_opening: str, target_moves_uci, proposed_uci_list):
    """Valuta la risposta dell'LLM con UNA sola metrica di sequenza completa,
    confrontabile con quella della GNN.

    `target_moves_uci` e' la linea COMPLETA e alternata (mossa del solver,
    risposta avversario, mossa del solver, ...). La GNN predice soltanto le
    PROPRIE mosse, quindi `full_sequence_correct` e' vero se la proposta
    coincide con la linea completa alternata OPPURE con le sole mosse del
    solver (indici pari della linea): in pratica "il modello ha trovato
    tutte le mosse che toccavano a lui". (In pratica i due formati danno gli
    stessi risultati, dato che il prompt chiede la linea alternata.)

    Metriche prodotte:
    - `first_move_correct`: la prima mossa coincide con quella di soluzione.
    - `full_sequence_correct`: vedi sopra (metrica primaria del confronto).
    - `proposed_legal`: la proposta e' legale (interpretata come linea
      alternata o come sole mosse del solver).
    - `interpretation`: quale formato ha usato il modello (diagnostica).
    """
    if not proposed_uci_list:
        return {"first_move_correct": False, "full_sequence_correct": False,
                "proposed_legal": False, "interpretation": "nessuna_risposta"}

    target_moves_uci = list(target_moves_uci)
    solver_targets = target_moves_uci[0::2]

    legal_alternating = _play_line(fen_after_opening, proposed_uci_list)
    legal_solver_only = (
        False if legal_alternating
        else _play_solver_only(fen_after_opening, proposed_uci_list, target_moves_uci)
    )

    if legal_alternating:
        interpretation = "linea_alternata"
    elif legal_solver_only:
        interpretation = "solo_mosse_solver"
    else:
        interpretation = "illegale"

    first_move_correct = (
        len(proposed_uci_list) > 0
        and len(target_moves_uci) > 0
        and proposed_uci_list[0] == target_moves_uci[0]
    )
    full_sequence_correct = (proposed_uci_list == target_moves_uci
                             or proposed_uci_list == solver_targets)

    return {
        "first_move_correct": first_move_correct,
        "full_sequence_correct": full_sequence_correct,
        "proposed_legal": legal_alternating or legal_solver_only,
        "interpretation": interpretation,
    }


def allocate_quota(group_sizes: dict, n_samples: int) -> dict:
    """Ripartisce `n_samples` fra i gruppi con un round-robin che
    REDISTRIBUISCE la quota non spendibile.

    Il campionamento precedente faceva `min(len(g), n_samples // n_gruppi)`
    per ogni gruppo: con 6 valori di n e n_samples=100 significava 16 a
    testa, ma i gruppi rari (n=5 con 1 solo puzzle, n=7 con 2) non
    riuscivano a spendere la loro quota e nessuno la raccoglieva. Da 100
    richiesti uscivano 46 puzzle effettivi. Qui invece la quota avanzata
    rifluisce sui gruppi che hanno ancora materiale, cosi' il campione
    finale e' davvero vicino a `n_samples` (i 100-200 chiesti dal PDF),
    restando il piu' bilanciato possibile fra i valori di n.
    """
    quota = {n: 0 for n in group_sizes}
    remaining = int(n_samples)
    while remaining > 0 and any(quota[n] < group_sizes[n] for n in group_sizes):
        for n in sorted(group_sizes):
            if remaining == 0:
                break
            if quota[n] < group_sizes[n]:
                quota[n] += 1
                remaining -= 1
    return quota


def load_evaluation_set(n_samples: int = config.LLM_N_SAMPLES) -> pd.DataFrame:
    """Carica il set di valutazione: l'held-out esterno (30 puzzle per ogni n
    da 1 a 5, vedi src/held_out.py). Con n_samples >= dimensione dell'held-out
    (default 150) si usa TUTTO l'held-out; con un valore minore si
    sottocampiona restando bilanciati per n.

    Se l'held-out non esiste si ripiega sul test set interno di Lichess,
    SOLO per collaudare la pipeline (non e' una valutazione valida).

    IL CAMPIONE E' PERSISTENTE (config.LLM_EVAL_SAMPLE_CSV): salvato la prima
    volta e ricaricato identico nelle riprese successive, cosi' una
    valutazione spalmata su piu' giorni (quota API) non mescola campioni
    diversi. Se il file salvato contiene puzzle che non esistono piu'
    nell'held-out attuale (campione di una versione precedente), viene
    scartato e rigenerato. `LLM_FRESH_START=1` forza la rigenerazione.
    """
    if config.LLM_FRESH_START and os.path.exists(config.LLM_EVAL_SAMPLE_CSV):
        os.remove(config.LLM_EVAL_SAMPLE_CSV)

    if os.path.exists(config.HELD_OUT_PARQUET):
        df_eval_full = pd.read_parquet(config.HELD_OUT_PARQUET)
        print(f"Uso il set held-out esterno: {len(df_eval_full):,} puzzle")
        print(df_eval_full.groupby(["mate_n", "source"]).size().rename("puzzle").to_string())
    elif os.path.exists(config.TEST_PARQUET):
        df_eval_full = pd.read_parquet(config.TEST_PARQUET)
        print(f"ATTENZIONE: held-out non trovato, uso il test set interno di Lichess "
              f"({len(df_eval_full):,} puzzle) solo per testare la pipeline.")
    else:
        raise FileNotFoundError("Nessun dataset di valutazione trovato. Esegui prima "
                                "`python -m scripts.run_pipeline`.")

    if os.path.exists(config.LLM_EVAL_SAMPLE_CSV):
        df_saved = pd.read_csv(config.LLM_EVAL_SAMPLE_CSV)
        if set(df_saved["PuzzleId"]) <= set(df_eval_full["PuzzleId"]):
            # Il campione resta lo stesso, ma le colonne di timing vanno prese
            # dall'held-out attuale (il timing proxy puo' essere cambiato).
            if "rating_norm" in df_eval_full.columns:
                df_saved = df_saved.drop(columns=[c for c in ("rating_norm", "simulated_think_time_sec")
                                                  if c in df_saved.columns])
                df_saved = df_saved.merge(df_eval_full[["PuzzleId", "rating_norm"]],
                                          on="PuzzleId", how="left")
            print(f"\nCampione di valutazione RIUSATO da {config.LLM_EVAL_SAMPLE_CSV} "
                  f"({len(df_saved)} puzzle). Distribuzione per n:")
            print(df_saved["mate_n"].value_counts().sort_index().to_string())
            return df_saved
        print(f"\nIl campione salvato in {config.LLM_EVAL_SAMPLE_CSV} contiene puzzle che "
              "non fanno parte dell'held-out attuale (versione precedente della pipeline): "
              "lo rigenero. Per evitare di mescolare risultati vecchi e nuovi, rilancia "
              "con LLM_FRESH_START=1 (sposta il vecchio llm_evaluation_raw.csv).")

    groups = {int(n): g for n, g in df_eval_full.groupby("mate_n")}
    quota = allocate_quota({n: len(g) for n, g in groups.items()}, n_samples)

    df_eval = pd.concat(
        [groups[n].sample(quota[n], random_state=0) for n in sorted(groups) if quota[n] > 0],
        ignore_index=True,
    )

    print(f"\nCampione di valutazione: {len(df_eval)} puzzle su {n_samples} richiesti:")
    for n in sorted(groups):
        print(f"  mate in {n}: {quota[n]:>3} scelti su {len(groups[n]):>3} disponibili")

    os.makedirs(os.path.dirname(config.LLM_EVAL_SAMPLE_CSV), exist_ok=True)
    df_eval.to_csv(config.LLM_EVAL_SAMPLE_CSV, index=False)
    print(f"Campione salvato in {config.LLM_EVAL_SAMPLE_CSV}.")
    return df_eval


def run_llm_evaluation(df_eval: pd.DataFrame, client, backend=config.LLM_BACKEND,
                        model_name=config.LLM_MODEL_NAME) -> pd.DataFrame:
    # --- Ripresa: se esiste già un CSV di risultati (da una run precedente
    # interrotta per quota esaurita), salta i puzzle già valutati con
    # successo invece di richiamarli di nuovo. ---
    already_done = pd.DataFrame()
    if config.LLM_FRESH_START and os.path.exists(config.LLM_RAW_CSV):
        backup = config.LLM_RAW_CSV.replace(".csv", "_precedente.csv")
        os.replace(config.LLM_RAW_CSV, backup)
        print(f"LLM_FRESH_START attivo: la run precedente e' stata spostata in {backup} "
              "e la valutazione riparte da zero (necessario dopo ogni modifica del "
              "prompt o delle metriche).")
    if os.path.exists(config.LLM_RAW_CSV):
        already_done = pd.read_csv(config.LLM_RAW_CSV)
        if "solver_moves_correct" in already_done.columns:
            # CSV di una versione con due metriche: ora ne esiste una sola e
            # corrisponde alla vecchia "permissiva".
            already_done["full_sequence_correct"] = already_done["solver_moves_correct"]
            already_done = already_done.drop(columns=["solver_moves_correct"])
        done_ids = set(already_done["PuzzleId"])
        n_before = len(df_eval)
        df_eval = df_eval[~df_eval["PuzzleId"].isin(done_ids)].reset_index(drop=True)
        skipped = n_before - len(df_eval)
        if skipped > 0:
            print(f"Ripresa: {skipped} puzzle già valutati in precedenza (in "
                  f"{config.LLM_RAW_CSV}), li salto. Restano {len(df_eval)} da fare.")

    llm_results = []
    quota_exceeded = False

    iterator = tqdm(
        df_eval.reset_index(drop=True).iterrows(),
        total=len(df_eval),
        desc="Valutazione LLM",
        unit="puzzle",
        leave=True,
        disable=(len(df_eval) == 0),
    )
    for i, row in iterator:
        fen_after, target_moves = get_position_after_opening(
            row["FEN"], row["Moves"],
            has_opening_move=row.get("has_opening_move", True),
        )
        prompt = build_prompt(fen_after, row["mate_n"])

        t0 = time.time()
        try:
            response_text = call_llm(client, prompt, backend=backend, model_name=model_name)
        except DailyQuotaExceededError:
            print(
                f"\nQuota giornaliera del modello '{model_name}' esaurita dopo "
                f"{i}/{len(df_eval)} puzzle di questa run "
                f"({len(llm_results)} riusciti). Mi fermo qui invece di sprecare "
                "altri tentativi che falliranno comunque fino al reset di domani.\n"
                "I risultati ottenuti finora sono già salvati. Rilancia lo stesso "
                "comando più tardi (o domani): riprenderà da dove si è fermato "
                "senza rifare le chiamate già riuscite."
            )
            quota_exceeded = True
            break
        except RuntimeError as e:
            print(f"[{i}] puzzle {row['PuzzleId']}: FALLITO ({e})")
            continue
        elapsed = time.time() - t0

        proposed = parse_solution(response_text, starting_fen=fen_after)
        eval_res = evaluate_llm_solution(fen_after, target_moves, proposed)

        llm_results.append({
            "PuzzleId": row["PuzzleId"],
            "mate_n": row["mate_n"],
            "fen_after_opening": fen_after,
            "target_moves": " ".join(target_moves),
            "proposed_moves": " ".join(proposed) if proposed else None,
            "response_time_sec": elapsed,
            "llm_explanation": response_text,
            **eval_res,
        })

        # Salvataggio incrementale: così anche un'interruzione a metà (quota,
        # timeout, chiusura del notebook) non perde il lavoro già fatto.
        df_so_far = pd.concat([already_done, pd.DataFrame(llm_results)], ignore_index=True)
        df_so_far.to_csv(config.LLM_RAW_CSV, index=False)

    df_llm = pd.concat([already_done, pd.DataFrame(llm_results)], ignore_index=True)
    df_llm.to_csv(config.LLM_RAW_CSV, index=False)

    if quota_exceeded:
        print(f"\nTotale accumulato finora: {len(df_llm)} puzzle valutati "
              f"(salvati in {config.LLM_RAW_CSV}).")
    else:
        print(f"\nCompletato: {len(df_llm)} puzzle valutati con successo in totale "
              f"({len(llm_results)} in questa run).")
    return df_llm


def print_llm_metrics(df_llm: pd.DataFrame) -> None:
    print("=== Metriche LLM (campione valutato) ===")
    print(f"Accuracy prima mossa:                  {df_llm['first_move_correct'].mean():.3f}")
    print(f"Accuracy sequenza completa:            {df_llm['full_sequence_correct'].mean():.3f}")
    print(f"% risposte con mosse legali:           {df_llm['proposed_legal'].mean():.3f}")
    print(f"Tempo medio di risposta (efficienza):  {df_llm['response_time_sec'].mean():.1f}s")
    if "interpretation" in df_llm.columns:
        print("\nFormato della risposta (diagnostica del prompt):")
        for k, v in df_llm["interpretation"].value_counts().items():
            print(f"  - {k}: {v} ({v / len(df_llm):.0%})")
    print("\nAccuracy per valore di n (prima mossa / sequenza completa):")
    print(df_llm.groupby("mate_n")[["first_move_correct", "full_sequence_correct"]].mean().round(3))
