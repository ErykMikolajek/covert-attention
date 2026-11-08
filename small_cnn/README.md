# Mała CNN: Posner, visual search, contextual cueing

Opis publikacji i protokołu: [ANALIZA.md](ANALIZA.md). Wyniki treningu: [results/RESULTS.md](results/RESULTS.md).

Jedna sieć, 142 366 parametrów, wejście 64×64, sześć klas. Model i generator w PyTorch/Pillow, bez jawnego attention. Materiały źródłowe w pozostałej części projektu pozostają bez zmian.

## Eksperyment lezji

Wyniki i kontrola interpretacji: [lesion_results/REPORT.md](lesion_results/REPORT.md).

```bash
python3 -m unittest discover -s small_cnn -p 'test_lesion.py' -v
python3 small_cnn/lesion.py
python3 small_cnn/lesion_attention.py
python3 small_cnn/lesion_checks.py
python3 small_cnn/lesion_readout_controls.py
MPLCONFIGDIR=tmp/matplotlib python3 small_cnn/summarize_lesion.py
```

`lesion.py` wybiera grupy na osobnej walidacji i ocenia je na 8000 nowych próbach na zadanie przy trzech poziomach szumu. Kontrole obejmują losowe grupy, zero i średnią aktywację, te same ID przy obu interwencjach, AUROC oraz diagnostyczne przekalibrowanie progu decyzji wyłącznie na walidacji. Żaden skrypt nie zmienia wag modelu.

## Uruchomienie z katalogu głównego projektu

```bash
python3 -m pip install -r small_cnn/requirements.txt
python3 -m unittest discover -s small_cnn -p 'test_*.py' -v
MPLCONFIGDIR=tmp/matplotlib python3 small_cnn/experiment.py --epochs 25
MPLCONFIGDIR=tmp/matplotlib python3 small_cnn/summarize.py
python3 small_cnn/diagnostics.py
python3 small_cnn/angle_oracle.py
python3 small_cnn/predict.py --index 0 --noise .04
```

Ponowna ewaluacja zapisanego checkpointu bez treningu:

```bash
MPLCONFIGDIR=tmp/matplotlib python3 small_cnn/experiment.py --evaluate
```

Powyższa ewaluacja nadpisuje pliki pomiarów, nie historię treningu ani checkpoint. Czas treningu jest odczytywany z istniejącego `history.json`.

Własny obraz o geometrii i jasności zgodnej z generatorem:

```bash
python3 small_cnn/predict.py --image /absolute/path/stimulus.png
python3 small_cnn/predict.py --image /absolute/path/stimulus.png --task posner
```

`--task` ogranicza decyzję do dwóch klas wybranego zadania. Bez niego sieć wybiera jedną z sześciu klas. Nie podawaj całego zrzutu ekranu z kilkoma panelami jako pojedynczego bodźca.

## Artefakty

- `results/best.pt`: wagi, konfiguracja, numer epoki, liczba parametrów.
- `results/config.json`, `history.json`: konfiguracja oraz log treningu i walidacji.
- `results/metrics.json`: accuracy, opisowe 95% CI, hit rate i false alarm rate, podział na zadania i warunki dla pięciu poziomów szumu.
- `results/test_predictions.jsonl`: predykcje z metadanymi próby.
- `results/posner_by_contrast.csv`, `diagnostics.json`: rozbicie hit/FAR po kontraście ramki, AUROC zadań i macierze pomyłek sześciu klas; powstają z `diagnostics.py`.
- `results/test_clean.npz`, `test_metadata.json`: 12 000 czystych bodźców testowych, etykiety i metadane. Szum odtwarza funkcja `predictions`.
- `results/stimuli.png`, `performance.png`, `attention_effects.png`: podgląd bodźców i wyników.
- `results/split_audit.json`: brak wspólnych nowych konfiguracji contextual cueing między splitami; wspólne czyste szablony są raportowane osobno.

Zbiory treningowe odtwarza generator z zapisanych seedów; nie wymagają dużego zewnętrznego datasetu. Testy sprawdzają bilans klas, brak kodowania orientacji przez stare konteksty, rozłączność nowych konfiguracji i interfejs sieci. Dodatkowe biblioteki PDF z `tmp/python_deps` służyły tylko do analizy publikacji i nie są zależnością modelu.
