"""
Módulo para evaluación y comparación de modelos de Machine Learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize


class ModelEvaluator:
    """
    Clase para evaluar y comparar múltiples modelos de clasificación o regresión.
    """

    def __init__(self, task: str = 'classification'):
        """
        Args:
            task: 'classification' o 'regression'
        """
        self.task = task
        self.results = []
        self.label_encoder = LabelEncoder()

    def evaluate_classifiers(
        self,
        modelos: Dict,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Entrena y evalúa múltiples clasificadores, devuelve tabla comparativa.

        Args:
            modelos: Diccionario {nombre: modelo}
            X_train, X_test: Features escaladas
            y_train, y_test: Labels codificadas
            cv: Folds para validación cruzada

        Returns:
            DataFrame con métricas comparativas
        """
        print("\n COMPARATIVA DE MODELOS DE CLASIFICACIÓN")
        

        # Codificar etiquetas si son strings
        if y_train.dtype == object:
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)

        n_classes = len(np.unique(y_train))
        resultados = []

        for nombre, modelo in modelos.items():
            print(f"\n Entrenando {nombre}...")

            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # AUC-ROC
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                y_test_bin = label_binarize(y_test, classes=range(n_classes))
                auc = roc_auc_score(
                    y_test_bin, y_proba,
                    average='weighted', multi_class='ovr'
                )

            # Cross-validation
            cv_scores = cross_val_score(
                modelo, X_train, y_train, cv=cv, scoring='accuracy'
            )

            resultados.append({
                'Modelo': nombre,
                'Accuracy (Test)': round(accuracy, 4),
                'F1-Score (Test)': round(f1, 4),
                'AUC-ROC (Test)': round(auc, 4),
                'CV Accuracy (Media)': round(cv_scores.mean(), 4),
                'CV Std': round(cv_scores.std(), 4)
            })

            print(f"    Accuracy: {accuracy:.4f} | F1: {f1:.4f} | "
                  f"AUC-ROC: {auc:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        df = pd.DataFrame(resultados).sort_values(
            'F1-Score (Test)', ascending=False
        ).reset_index(drop=True)

        print("\n\n TABLA COMPARATIVA")
        
        print(df.to_string(index=False))
        

        self.results = df
        return df

    def evaluate_regressors(
        self,
        modelos: Dict,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Entrena y evalúa múltiples regresores, devuelve tabla comparativa.

        Args:
            modelos: Diccionario {nombre: modelo}
            X_train, X_test: Features escaladas
            y_train, y_test: Valores objetivo
            cv: Folds para validación cruzada

        Returns:
            DataFrame con métricas comparativas
        """
        print("\n COMPARATIVA DE MODELOS DE REGRESIÓN")
        

        resultados = []

        for nombre, modelo in modelos.items():
            print(f"\n Entrenando {nombre}...")

            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            cv_scores = cross_val_score(
                modelo, X_train, y_train, cv=cv, scoring='r2'
            )

            resultados.append({
                'Modelo': nombre,
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'R² (Test)': round(r2, 4),
                'CV R² (Media)': round(cv_scores.mean(), 4),
                'CV Std': round(cv_scores.std(), 4)
            })

            print(f"   MAE: {mae:.4f} | RMSE: {rmse:.4f} | "
                  f"R²: {r2:.4f} | CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        df = pd.DataFrame(resultados).sort_values(
            'R² (Test)', ascending=False
        ).reset_index(drop=True)

        print("\n\n TABLA COMPARATIVA")
        
        print(df.to_string(index=False))
        

        self.results = df
        return df

    def plot_comparison(
        self,
        metricas: List[str] = None,
        title: str = 'Comparativa de Modelos'
    ):
        """
        Visualiza la comparativa de modelos en barras.

        Args:
            metricas: Lista de columnas a visualizar
            title: Título del gráfico
        """
        if not len(self.results):
            raise ValueError("Ejecuta evaluate_classifiers o evaluate_regressors primero")

        if metricas is None:
            if self.task == 'classification':
                metricas = ['Accuracy (Test)', 'F1-Score (Test)', 'AUC-ROC (Test)']
            else:
                metricas = ['MAE', 'RMSE', 'R² (Test)']

        colores = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple',
                   'darkorange', 'crimson']

        fig, axes = plt.subplots(1, len(metricas), figsize=(6 * len(metricas), 6))

        if len(metricas) == 1:
            axes = [axes]

        for idx, metrica in enumerate(metricas):
            bars = axes[idx].bar(
                self.results['Modelo'],
                self.results[metrica],
                color=colores[:len(self.results)],
                edgecolor='black',
                linewidth=0.8,
                alpha=0.85
            )
            axes[idx].set_title(metrica, fontsize=13, fontweight='bold')
            axes[idx].set_ylabel('Valor', fontsize=11)
            axes[idx].set_xticklabels(
                self.results['Modelo'], rotation=20, ha='right'
            )
            axes[idx].grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, self.results[metrica]):
                axes[idx].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold'
                )

        plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    def print_winner(self, metric: str = None):
        """
        Imprime el modelo ganador y justificación.
        """
        if not len(self.results):
            raise ValueError("Ejecuta evaluate_classifiers o evaluate_regressors primero")

        if metric is None:
            metric = 'F1-Score (Test)' if self.task == 'classification' else 'R² (Test)'

        ganador = self.results.iloc[0]['Modelo']

        print(f"\n MODELO GANADOR: {ganador}")
        
        print(self.results.iloc[0].to_string())
        

        return ganador