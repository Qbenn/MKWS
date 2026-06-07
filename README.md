# Metody numeryczne w modelowaniu spalania mieszanki metan–powietrze

Projekt na przedmiot **Metody Komputerowe w Spalaniu (MKWS)**, Politechnika
Warszawska, Wydział Mechaniczny Energetyki i Lotnictwa.

Samodzielna implementacja trzech klas metod numerycznych zastosowanych do
modelowania spalania metanu (CH₄). Celowo zrezygnowano z gotowych pakietów
kinetycznych (np. Cantera), aby od podstaw zaimplementować i zweryfikować same
metody numeryczne.

## Zakres projektu

| Część | Zagadnienie | Metoda numeryczna |
|-------|-------------|-------------------|
| 1 | Adiabatyczna temperatura płomienia *T*<sub>ad</sub>(φ) | Newton–Raphson (równanie nieliniowe) |
| 2 | Zapłon w reaktorze 0D, *T*(*t*), *Y*(*t*), wykres Arrheniusa | Runge–Kutta 4. rzędu (układ ODE) |
| 3 | Profil płomienia laminarnego 1D | różnice skończone + Newton + algorytm Thomasa (PDE) |

## Struktura plików

```
projekt_spalanie_mkws/
├── thermo.py          # dane NASA, własności mieszaniny, stechiometria
├── flame_temp.py      # T_ad metodą Newtona–Raphsona
├── reactor0d.py       # reaktor 0D, kinetyka Arrheniusa, całkowanie RK4
├── flame1d.py         # płomień laminarny 1D, różnice skończone, algorytm Thomasa
├── main.py            # uruchamia wszystkie części i generuje wykresy
├── figures/           # wygenerowane wykresy (PNG) używane w raporcie
├── raport.tex         # źródło raportu LaTeX
├── raport.pdf         # skompilowany raport
├── requirements.txt   # zależności Pythona
└── README.md
```

## Wymagania

- Python 3.8+
- biblioteki wymienione w `requirements.txt` (NumPy, Matplotlib)
- do kompilacji raportu: dystrybucja LaTeX z pakietem `babel-polish`
  (np. TeX Live) lub konto na Overleaf

## Uruchomienie

### Część obliczeniowa

```bash
pip install -r requirements.txt
python3 main.py
```

Skrypt wykona wszystkie trzy części i zapisze wykresy do katalogu `figures/`.
W konsoli wypisane zostaną kluczowe wyniki, m.in. *T*<sub>ad</sub>, czas zapłonu
oraz grubość płomienia.

### Kompilacja raportu

```bash
pdflatex raport.tex
pdflatex raport.tex
```

(dwukrotnie — za drugim razem generowany jest spis treści). Alternatywnie można
wgrać `raport.tex` wraz z katalogiem `figures/` na Overleaf.

## Główne wyniki

- **Część 1:** *T*<sub>ad</sub> ≈ 2326 K dla mieszanki stechiometrycznej (φ = 1);
  metoda Newtona zbiega kwadratowo w 3–4 iteracjach.
- **Część 2:** odtworzono charakterystyczny przebieg samozapłonu oraz zależność
  typu Arrheniusa czasu zapłonu od temperatury (pozorna energia aktywacji
  ≈ 175 kJ/mol).
- **Część 3:** gładki profil temperatury z wyraźną strefą reakcji; wykazano
  zbieżność siatkową rozwiązania.

## Ograniczenia modelu

Zastosowano jednokrokową kinetykę globalną i pominięto dysocjację produktów, co
zawyża temperatury końcowe względem wartości pomiarowych. Naturalnym kierunkiem
rozszerzenia jest użycie wieloetapowego mechanizmu reakcji (np. GRI-Mech 3.0),
adaptacyjnego kroku czasowego dla sztywnego układu kinetycznego oraz wyznaczanie
laminarnej prędkości spalania *S*<sub>u</sub> jako zagadnienia własnego.
