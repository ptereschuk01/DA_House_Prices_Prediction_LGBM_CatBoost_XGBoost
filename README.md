# Immobilienpreisprognose mit Gradient Boosting

Vergleich von LightGBM, CatBoost und XGBoost auf dem Kaggle-Datensatz **House Prices — Advanced Regression Techniques**. Ziel ist, die Datenaufbereitung, das Verhalten unterschiedlicher Modelle und mögliche Ensemble-Ansätze nachvollziehbar zu untersuchen.

**Status:** Die Analyse wird weiterentwickelt. Das Repository enthält Notebook, Ausgangsdaten und eine erzeugte Submission-Datei.

[Analyse-Notebook öffnen](notebooks/House_7_1_Prices_Prediction_LGBM_CatBoost_XGBoost.ipynb)

## Vorgehen

1. Trainingsdaten laden und numerische sowie kategoriale Merkmale unterscheiden.
2. Fehlende Werte behandeln und Kategorien für LightGBM/XGBoost per One-Hot-Encoding aufbereiten.
3. Die Zielgröße `SalePrice` mit `log1p` transformieren.
4. LightGBM, CatBoost und XGBoost mit fünf Validierungsfaltungen vergleichen.
5. Merkmalsbedeutung und ausgewählte Hyperparameter untersuchen.
6. Einfache und gewichtete Modellkombinationen erproben und Vorhersagen exportieren.

## Gespeicherte Ergebnisse

Die folgende Tabelle übernimmt die im Notebook gespeicherten Ausgaben der fünffachen Kreuzvalidierung. Die Werte wurden für diese Dokumentationsüberarbeitung nicht neu berechnet.

| Modell | Mittlerer RMSE auf log1p(SalePrice) |
| --- | ---: |
| CatBoost | 0.1274 |
| LightGBM | 0.1356 |
| XGBoost | 0.1404 |

Die Werte beziehen sich auf die logarithmierte Zielgröße, nicht auf einen Fehlerbetrag in Euro oder Dollar.

### Einordnung der Validierung

Bei LightGBM und XGBoost erfolgt die Vorverarbeitung derzeit vor der Aufteilung in Validierungsfaltungen. Damit fließen Informationen über die Verteilung der späteren Validierungsdaten in die Vorverarbeitung ein. Für einen belastbaren Vergleich muss die Vorverarbeitung innerhalb jeder Trainingsfaltung angepasst werden.

Die später im Notebook ausgegebenen Ensemble-RMSE werden auf den Trainingsdaten berechnet. Sie sind kein Nachweis für die Vorhersagequalität auf unbekannten Daten. Ein Vergleich anhand von Out-of-Fold-Vorhersagen bleibt offen.

## Dateien

| Pfad | Inhalt |
| --- | --- |
| `data/raw/` | Trainings-, Test- und Beispieldaten |
| `docs/data_description.txt` | Beschreibung der Merkmale |
| `notebooks/` | Analyse und gespeicherte Submission |
| `requirements.txt` | Auflistung der Python-Abhängigkeiten |

## Lokal öffnen

Nach dem Klonen des Repositorys eine Python-Umgebung aktivieren und die Abhängigkeiten installieren:

```bash
python -m pip install -r requirements.txt
python -m pip install xgboost
python -m notebook
```

XGBoost wird vom Notebook benötigt, fehlt aber derzeit in `requirements.txt`. Die dort genannten Pakete sind nicht auf bestimmte Versionen festgelegt.

Das Notebook im Verzeichnis `notebooks/` öffnen. Die vorhandenen Datenpfade verwenden Windows-Schreibweise; für andere Betriebssysteme sind sie anzupassen.

## Nächste fachliche Schritte

- Vorverarbeitung innerhalb der jeweiligen Validierungsfaltung ausführen.
- Trainings- und Testtransformationen konsistent über dieselben angepassten Verarbeitungsschritte anwenden.
- Ensembles mit Out-of-Fold-Vorhersagen beurteilen.
- Umgebung und endgültigen Modellvergleich reproduzierbar dokumentieren.

**Technologien:** Python · pandas · NumPy · scikit-learn · LightGBM · CatBoost · XGBoost · matplotlib · seaborn

[Gesamtes Portfolio](https://github.com/ptereschuk01)
