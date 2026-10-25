"""
src/stats.py
-------------
Test statistici appaiati condivisi, usati in due punti del progetto:

- src/compare.py  -> confronto GNN vs LLM (obiettivo 4 del PDF)
- src/train.py    -> ablation timing vs no-timing (obiettivo 5 del PDF)

Il PDF chiede esplicitamente "t-test accoppiati o il test di McNemar".
Qui si usa McNemar, che e' la scelta corretta per esiti BINARI appaiati
(puzzle risolto / non risolto dallo stesso identico insieme di puzzle),
mentre il t-test accoppiato sarebbe appropriato per misure continue.

Due varianti calcolate SEMPRE insieme:

1. chi-quadro con correzione di continuita' (la formula "classica"), valida
   in modo affidabile solo quando b + c >= 25;
2. test binomiale ESATTO, valido per qualunque numero di discordanze.

Con i numeri tipici di questo progetto (poche decine di puzzle valutati con
l'LLM) b + c e' quasi sempre piccolo, quindi il numero da RIPORTARE NEL
REPORT e' il p-value esatto. Il chi-quadro resta stampato solo per
continuita' con la formulazione del PDF.
"""

from math import comb


def mcnemar_from_counts(b: int, c: int) -> dict:
    """McNemar su una tabella 2x2 appaiata.

    `b` = casi in cui il modello 1 e' corretto e il modello 2 no
    `c` = casi in cui il modello 2 e' corretto e il modello 1 no
    (le celle concordanti non entrano nel test: e' proprio il punto di
    McNemar, che guarda solo i disaccordi).
    """
    b, c = int(b), int(c)
    n = b + c
    result = {
        "b": b, "c": c, "n_discordant": n,
        "chi2": None, "p_chi2": None, "p_exact": None,
    }
    if n == 0:
        return result

    # 1) chi-quadro con correzione di continuita'
    from scipy.stats import chi2 as chi2_dist
    stat = (abs(b - c) - 1) ** 2 / n
    result["chi2"] = stat
    result["p_chi2"] = float(chi2_dist.sf(stat, df=1))

    # 2) binomiale esatto a due code, H0: p = 0.5 su n discordanze
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    result["p_exact"] = float(min(1.0, 2 * tail))
    return result


def print_mcnemar(result: dict, label_1: str, label_2: str,
                  both_correct=None, neither=None, alpha: float = 0.05) -> None:
    """Stampa leggibile della tabella e dei due p-value."""
    b, c = result["b"], result["c"]
    if both_correct is not None and neither is not None:
        print(f"                     {label_2} OK   {label_2} KO")
        print(f"{label_1} OK       {both_correct:^12}  {b:^12}")
        print(f"{label_1} KO       {c:^12}  {neither:^12}")

    if result["n_discordant"] == 0:
        print("\nNessuna discordanza (b + c = 0): il test di McNemar non e' applicabile "
              "su questo campione — i due modelli hanno sbagliato e indovinato esattamente "
              "gli stessi puzzle.")
        return

    print(f"\nDiscordanze: b = {b} (solo {label_1}), c = {c} (solo {label_2}), "
          f"b + c = {result['n_discordant']}")
    print(f"McNemar chi2 (con correzione di continuita') = {result['chi2']:.3f}, "
          f"p = {result['p_chi2']:.4f}")
    print(f"McNemar binomiale ESATTO                     p = {result['p_exact']:.4f}  "
          "<-- da riportare nel report")
    if result["n_discordant"] < 25:
        print(f"    (b + c = {result['n_discordant']} < 25: l'approssimazione chi-quadro non e' "
              "affidabile qui, usare il p-value esatto)")

    p = result["p_exact"]
    if p < alpha:
        winner = label_1 if b > c else label_2
        print(f"-> Differenza statisticamente significativa (p < {alpha}): "
              f"{winner} prevale su questo campione.")
    else:
        print(f"-> Nessuna differenza statisticamente significativa (p >= {alpha}) "
              "su questo campione. Con pochi puzzle valutati questo e' un esito "
              "atteso anche quando una differenza reale esiste: aumentare la "
              "numerosita' prima di concludere alcunche'.")


def mcnemar_from_flags(flags_1, flags_2, label_1: str, label_2: str,
                        verbose: bool = True) -> dict:
    """Come sopra, ma a partire da due sequenze booleane appaiate (stessa
    lunghezza, stesso ordine di puzzle)."""
    import numpy as np
    a1 = np.asarray(list(flags_1), dtype=bool)
    a2 = np.asarray(list(flags_2), dtype=bool)
    if len(a1) != len(a2):
        raise ValueError("Le due sequenze devono essere appaiate (stessa lunghezza).")

    both = int((a1 & a2).sum())
    b = int((a1 & ~a2).sum())
    c = int((~a1 & a2).sum())
    neither = int((~a1 & ~a2).sum())

    res = mcnemar_from_counts(b, c)
    res.update({"both_correct": both, "neither": neither, "n_pairs": len(a1)})
    if verbose:
        print_mcnemar(res, label_1, label_2, both_correct=both, neither=neither)
    return res
