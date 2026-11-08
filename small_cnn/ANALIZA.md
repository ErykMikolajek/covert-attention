# Cel projektu i interpretacja prac Srivastavy i współautorów

## Dwie publikacje, dwa poziomy wyjaśnienia

**PNAS 2025, „Emergent neuronal mechanisms mediating covert attention in convolutional neural networks”** pyta, jak zwykła sieć feedforward, optymalizowana do detekcji celu, wytwarza efekt wskazówki przestrzennej bez jawnie zaprogramowanego attention. Przedmiotem pracy nie jest zwiększanie skuteczności klasyfikatora ani implementacja transformerowego attention. Autorzy używają CNN jako układu, w którym mogą zmierzyć odpowiedź każdej jednostki i prześledzić drogę od pikseli do decyzji.

Głównym eksperymentem jest Posner: dwa odcinki, wskazówka w postaci ramki i decyzja „cel obecny/nieobecny”. Cel występuje w połowie prób, a przy jego obecności ramka poprawnie wskazuje położenie w 80% prób. W tej publikacji kąt celu wynosi średnio 14°, dystraktora 7°, z odchyleniem standardowym 3,5°. Szum orientacji sprawia, że klasy nakładają się: nawet idealne odczytanie obrazu nie gwarantuje 100% trafności. Autorzy porównują sieci uczone z trafnością wskazówki 80% i 50%.

Analiza obejmuje 10 sieci i łącznie około 1,8 mln jednostek aktywacji, a nie 1,8 mln niezależnych parametrów. Metody obejmują AUROC pojedynczych jednostek, badanie odpowiedzi na sam cel, samą wskazówkę i ich kombinacje oraz reverse correlation/regresję do szacowania pól recepcyjnych. To rozróżnienie jest istotne: AUROC neuronu i accuracy całej sieci odpowiadają na inne pytania.

Autorzy znajdują wczesne jednostki selektywne osobno na cel i wskazówkę, następnie wspólne kodowanie celu i wskazówki, a dalej integrację informacji z obu lokalizacji. Opisują cztery mechanizmy: sumowanie informacji z lokalizacji z wpływem wskazówki, przeciwstawianie lokalizacji, ich kombinację oraz interakcję odpowiedzi na wskazówkę z progiem ReLU. Ta ostatnia może zmieniać rozróżnialność celu i dystraktora przez przesuwanie rozkładu aktywacji względem zera, bez dodatkowego modułu gain.

Reanaliza danych ze wzgórka górnego myszy wskazuje komórki o niektórych przewidywanych właściwościach, m.in. hamowaniu związanym ze wskazówką oraz sumowaniu i przeciwstawianiu lokalizacji. Nie wszystkie przewidywane typy znaleziono. Podobieństwo CNN do danych neuronalnych nie dowodzi, że mózg używa dokładnie tego samego algorytmu. Sami autorzy wskazują efekty, których model nie wyjaśnia: m.in. pełny przebieg czasowy uwagi i niektóre efekty niewskazujących predykcyjnie bodźców egzogennych.

**`mmc2.pdf` to pełna praca Current Biology 2024 „Emergent human-like covert attention in feedforward convolutional neural networks” wraz z metodami i suplementem**, a nie suplement neuronalny do artykułu PNAS. Jej celem jest wyjaśnienie zachowania w trzech paradygmatach. Sieci uczone z pikseli i etykiet mogą wykorzystywać statystyki wskazówek i kontekstów bez specjalnego zasobu uwagi. Autorzy porównują je z ludźmi i bayesowskim obserwatorem idealnym (BIO). Standardowo każda sieć jest trenowana osobno do danego zadania, ale metody opisują też wspólny model z sześcioma klasami.

## Co oznaczają trzy zadania

| Zadanie | Co przewiduje model | Co jest manipulowane | Właściwy efekt behawioralny |
|---|---|---|---|
| Posner | Obecność bardziej pochylonego odcinka | Poprawność i kontrast ramki | Wyższy hit rate dla valid niż invalid przy kontrolowanym false alarm rate |
| Visual search | Obecność pochylonego odcinka wśród ośmiu elementów | Liczba ramek: 1, 2, 4, 8 | Pogorszenie wykrywania przy wzroście liczby dopuszczalnych lokalizacji |
| Contextual cueing | T skierowane w lewo albo w prawo, wśród L | Powtarzany albo nowy układ tła i lokalizacji celu | Wyższa trafność dla starych układów przy niezależnym szumie |

W visual search **set size oznacza liczbę wskazanych ramek, nie liczbę odcinków na obrazie**: odcinków zawsze jest osiem. Cel, jeśli istnieje, znajduje się w jednej z ramek. W contextual cueing „old/new” nie jest etykietą klasyfikacyjną. Model ma odczytać orientację T. Stare tło informuje, gdzie szukać T, ale nie może informować, w którą stronę T jest skierowane.

Duża trafność to inny wynik niż odtworzenie efektu uwagowego. Na łatwych obrazach można osiągnąć niemal 100% i nie zobaczyć różnicy old/new z powodu efektu sufitu. Należy raportować oba wyniki, przy kilku poziomach szumu. Jednorazowe przejście 2D CNN nie daje biologicznego czasu reakcji; nie przeliczamy czasu wykonania programu na czas reakcji człowieka.

Dodatkowy punkt odniesienia dla naszej wersji Posnera: obserwator mający bezszumowy dostęp do prawdziwych kątów oraz strony ramki osiąga w symulacji miliona prób około **84,64%**. Obliczamy iloraz wiarygodności rozkładów N(15; 3,5²) i N(7; 3,5²), sumujemy go z wagami 0,8/0,2 i porównujemy z 1. To optymistyczny benchmark na ukrytych kątach, a nie obrazowy BIO z publikacji. Ignoruje szum luminancji i rasteryzację, a nawet zakłada znajomość strony niewidocznej ramki przy zerowym kontraście. Pokazuje, dlaczego około 80% w Posnerze może oznaczać dobry wynik. Skrypt `angle_oracle.py` i `results/angle_oracle.json` pozwalają odtworzyć obliczenie.

## Co rzeczywiście było w projekcie

- `Code/`: notebooki z generatorami Posnera, treningiem i analizą AUROC; m.in. `Posner_80%_stimgen_training_AUC.ipynb`.
- `Model Weights/`: dwa archiwa z dziesięcioma modelami HDF5 dla wskazówek 80% i 50%.
- `CSV Files/`: wyniki analiz, m.in. trafności i AUROC jednostek. To nie zbiór obrazów z etykietami do trzech zadań.
- `Stimuli/`: pusty katalog w momencie inspekcji.
- `Shapely_perturbations/`: istniejące analizy i wyniki perturbacji; nie zostały zmodyfikowane.

W związku z tym nowy model jest uczony od zera na **nowo wygenerowanych bodźcach**, opartych na geometrii i statystykach notebooków Posnera oraz opisie trzech zadań w `mmc2.pdf`. Nie jest to trening na istniejącym gotowym datasecie trzech zadań ani fine-tuning dostarczonych wag. Wagi i CSV pozostają materiałem do porównań oraz dalszej analizy neuronalnej.

## Implementacja i świadome odstępstwa od publikacji

Jedna CNN przyjmuje tylko obraz w skali szarości 64×64. Ma trzy bloki Conv–ReLU–MaxPool (16, 24, 32 kanały), warstwę dense 64 i sześć logitów: Posner absent/present, search absent/present, context left/right. Łącznie **142 366 parametrów**. Nie dostaje współrzędnych celu, rodzaju wskazówki ani flagi old/new. Funkcja kosztu to zwykła cross-entropy sześciu klas. Dodatkowo raportujemy trafność przy zadanym zadaniu przez ograniczenie decyzji do odpowiednich dwóch klas; główny wynik to decyzja spośród wszystkich sześciu.

Rozdzielczość i architektura są adaptacją do małego modelu CPU. Rysowanie odbywa się w trzykrotnie większej rozdzielczości z późniejszym antyaliasingiem. Kąty Posnera to 15° i 7° z SD 3,5°, zgodnie z pracą 2024; notebook/PNAS 2025 używa 14° dla celu. Ramki Posnera mają przyrost jasności 0, 20, 40 albo 60 względem 128. Search ma odcinki 0° i 15°, bez szumu orientacji, osiem stałych pozycji i konfiguracje ramek zgodne z opisem metod. Context ma 16 pozycji na siatce, jeden T i siedem L; zdefiniowano 16 starych kontekstów, po jednym na pozycję celu. Nowe układy są generowane niezależnie; ich dwie orientacje nie muszą tworzyć par o identycznym tle, w odróżnieniu od procedury publikacji. To jawne uproszczenie generatora.

Obrazy normalizujemy jako `(pixel-128)/128`, a następnie dodajemy niezależny szum Gaussa. Nie obcinamy zaszumionych wartości do zakresu obrazu. W treningu poziom SD jest losowany spośród 0,02; 0,04; 0,08; 0,14. Główna walidacja używa z góry ustalonego SD=0,04; test obejmuje SD=0; 0,04; 0,08; 0,14; 0,20. Ze względu na inną rasteryzację i rozdzielczość ta sama liczba SD nie oznacza dokładnie takiej samej trudności jak w publikacji. Nie dopasowujemy szumu do wyników ludzi.

Zbiory mają odpowiednio 10 000, 2 000 i 4 000 obrazów **na każde zadanie**, czyli 30 000/6 000/12 000 ogółem. Klasy są zbalansowane. Seed główny wynosi 42, generatory splitów 142/242/342. Każda epoka dodaje nowy szum do czystych obrazów treningowych. Stare układy contextual cueing z definicji powtarzają się między splitami; orientacja T jest w nich dokładnie zbalansowana. Nowe konfiguracje nie powtarzają się między splitami, co sprawdza audyt. Czyste, skwantowane bodźce mogą się powtarzać w Posner/search: te zadania używają ograniczonego zestawu geometrii, a próby różnią się niezależnym szumem. Test nie służy do wyboru epoki.

AdamW, learning rate 0,001 z harmonogramem cosine do 0,0001, batch 128, weight decay 0,0001. Zapisujemy checkpoint o najwyższej trafności na walidacji. Wyniki to jeden trening, nie średnia z dziesięciu niezależnych sieci jak w publikacjach. Przedziały Wilsona opisują niepewność prób dla jednej sieci i nie obejmują zmienności inicjalizacji ani zależności wynikających z powtarzanych starych kontekstów. Nie są podstawą do wniosków o istotności mechanizmu uwagi.

## Źródła

1. Lokalny PDF PNAS 2025: `papers/srivastava-et-al-2025-emergent-neuronal-mechanisms-mediating-covert-attention-in-convolutional-neural-networks.pdf`, szczególnie strony 1–3, 6, 9–10. DOI: https://doi.org/10.1073/pnas.2411909122.
2. Lokalny `papers/mmc2.pdf`, Figure 1 oraz STAR Methods: „Stimuli and psychophysics”, „Training details of CNN”, „CNN architecture for joint training of the three tasks”, „Training a smaller size network”. DOI: https://doi.org/10.1016/j.cub.2023.12.058.
3. Korekta PNAS opublikowana 9 stycznia 2026 dotyczy kolorów etykiet Fig. 6H, nie zmiany definicji zadań ani wyników treningu: https://doi.org/10.1073/pnas.2536323123.

Wyniki wykonanej sesji, warunki testowe i ograniczenia znajdują się w `results/RESULTS.md`; surowe pomiary w `results/metrics.json` i `results/test_predictions.jsonl`.
