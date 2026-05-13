"""
Módulo para modelos clásicos de Machine Learning (Random Forest, SVM, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class TacticalStyleClassifier:
    """
    Clasificador de estilos tácticos usando modelos clásicos de ML.
    """
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Inicializa el clasificador.
        
        Args:
            model_type: Tipo de modelo ('random_forest', 'gradient_boosting', 'svm')
            random_state: Semilla para reproducibilidad
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False
        
        # Inicializar modelo
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Inicializa el modelo según el tipo especificado.
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Modelo '{self.model_type}' no soportado")
        
        print(f" Modelo inicializado: {self.model_type}")
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'label',
        exclude_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos para entrenamiento.
    
        Args:
            df: DataFrame con features y etiquetas
            target_col: Nombre de la columna objetivo
            exclude_cols: Columnas a excluir de features
        
        Returns:
            Tupla (X, y)
        """
        if exclude_cols is None:
            # Excluir columnas por defecto
            exclude_cols = [
                'match_id', 'team', target_col,
                # Excluir labels de otros targets
                'resultado', 'goles_equipo', 'goles_rival',
                'xg_equipo', 'xg_rival', 'xg_diff',
                # Excluir columnas categóricas
                'contexto', 'rival', 'tipo_rival', 'es_barcelona'
            ]
    
        # Separar features y target
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[target_col]
    
        # Guardar nombres de features
        self.feature_names = X.columns.tolist()
    
        # Verificar que solo quedan columnas numéricas
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"  Eliminando columnas no numéricas: {non_numeric}")
            X = X.select_dtypes(include=[np.number])
            self.feature_names = X.columns.tolist()
    
        # Manejar valores faltantes
        if X.isnull().sum().sum() > 0:
            print(f"  Imputando {X.isnull().sum().sum()} valores faltantes con la media")
            X = X.fillna(X.mean())
    
        print(f" Datos preparados:")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Instancias: {X.shape[0]}")
        print(f"   - Clases: {y.nunique()}")
        print(f"   - Distribución: {y.value_counts().to_dict()}")
    
        return X, y
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        scale: bool = True
    ):
        """
        Entrena el modelo.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            scale: Escalar features (recomendado para SVM)
        """
        print(f"\n Entrenando modelo {self.model_type}...")
        
        # Codificar etiquetas si son strings
        if y_train.dtype == 'object':
            y_train = self.label_encoder.fit_transform(y_train)
        
        # Escalar features
        if scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
        
        # Entrenar
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Accuracy en training
        train_accuracy = self.model.score(X_train_scaled, y_train)
        
        print(f" Modelo entrenado")
        print(f"   - Accuracy (train): {train_accuracy:.4f}")
    
    def predict(self, X_test: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """
        Realiza predicciones.
        
        Args:
            X_test: Features de test
            scale: Escalar features
            
        Returns:
            Array con predicciones
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecuta .train() primero")
        
        if scale:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        predictions = self.model.predict(X_test_scaled)
        
        # Decodificar si es necesario
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X_test: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """
        Predice probabilidades.
        
        Args:
            X_test: Features de test
            scale: Escalar features
            
        Returns:
            Array con probabilidades por clase
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Modelo {self.model_type} no soporta predict_proba")
        
        if scale:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        return self.model.predict_proba(X_test_scaled)
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        scale: bool = True
    ) -> Dict:
        """
        Evalúa el modelo en datos de test.
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            scale: Escalar features
            
        Returns:
            Diccionario con métricas
        """
        print(f"\n Evaluando modelo...")
        
        # Codificar etiquetas si es necesario
        if y_test.dtype == 'object':
            y_test_encoded = self.label_encoder.transform(y_test)
        else:
            y_test_encoded = y_test
        
        # Predicciones
        predictions = self.predict(X_test, scale=scale)
        
        if hasattr(self.label_encoder, 'classes_'):
            predictions_encoded = self.label_encoder.transform(predictions)
        else:
            predictions_encoded = predictions
        
        # Métricas
        accuracy = accuracy_score(y_test_encoded, predictions_encoded)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_encoded, predictions_encoded, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # AUC-ROC (solo para clasificación binaria o con predict_proba)
        if hasattr(self.model, 'predict_proba') and len(np.unique(y_test_encoded)) == 2:
            proba = self.predict_proba(X_test, scale=scale)[:, 1]
            metrics['auc_roc'] = roc_auc_score(y_test_encoded, proba)
        
        print(f" Resultados:")
        print(f"   - Accuracy: {metrics['accuracy']:.4f}")
        print(f"   - Precision: {metrics['precision']:.4f}")
        print(f"   - Recall: {metrics['recall']:.4f}")
        print(f"   - F1-Score: {metrics['f1_score']:.4f}")
        if 'auc_roc' in metrics:
            print(f"   - AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv: int = 5,
        scale: bool = True
    ) -> Dict:
        """
        Realiza validación cruzada.
        
        Args:
            X: Features
            y: Labels
            cv: Número de folds
            scale: Escalar features
            
        Returns:
            Diccionario con resultados de CV
        """
        print(f"\n Validación cruzada (cv={cv})...")
        
        # Codificar etiquetas
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Escalar
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Cross-validation
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        results = {
            'cv_scores': scores,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }
        
        print(f" Resultados CV:")
        print(f"   - Accuracy promedio: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
        print(f"   - Scores por fold: {scores}")
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Obtiene importancia de features (solo para modelos basados en árboles).
        
        Args:
            top_n: Número de features más importantes a retornar
            
        Returns:
            DataFrame con features e importancia
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"Modelo {self.model_type} no tiene feature_importances_")
        
        importances = self.model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20):
        """
        Visualiza importancia de features.
        
        Args:
            top_n: Número de features a mostrar
        """
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
        plt.xlabel('Importancia')
        plt.title(f'Top {top_n} Features Más Importantes - {self.model_type.title()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        scale: bool = True
    ):
        """
        Visualiza matriz de confusión.
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            scale: Escalar features
        """
        # Codificar si es necesario
        if y_test.dtype == 'object':
            y_test_encoded = self.label_encoder.transform(y_test)
            labels = self.label_encoder.classes_
        else:
            y_test_encoded = y_test
            labels = np.unique(y_test)
        
        # Predicciones
        predictions = self.predict(X_test, scale=scale)
        
        if hasattr(self.label_encoder, 'classes_'):
            predictions_encoded = self.label_encoder.transform(predictions)
        else:
            predictions_encoded = predictions
        
        # Matriz de confusión
        cm = confusion_matrix(y_test_encoded, predictions_encoded)
        
        # Visualizar
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusión - {self.model_type.title()}')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        scale: bool = True
    ):
        """
        Visualiza curva ROC (solo clasificación binaria).
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            scale: Escalar features
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Modelo no soporta predict_proba")
        
        # Codificar
        if y_test.dtype == 'object':
            y_test_encoded = self.label_encoder.transform(y_test)
        else:
            y_test_encoded = y_test
        
        if len(np.unique(y_test_encoded)) != 2:
            raise ValueError("ROC curve solo disponible para clasificación binaria")
        
        # Probabilidades
        proba = self.predict_proba(X_test, scale=scale)[:, 1]
        
        # Calcular ROC
        fpr, tpr, thresholds = roc_curve(y_test_encoded, proba)
        auc = roc_auc_score(y_test_encoded, proba)
        
        # Visualizar
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Azar')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {self.model_type.title()}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict = None,
        cv: int = 5,
        scale: bool = True
    ) -> Dict:
        """
        Búsqueda de hiperparámetros óptimos.
        
        Args:
            X: Features
            y: Labels
            param_grid: Grilla de parámetros a probar
            cv: Número de folds
            scale: Escalar features
            
        Returns:
            Diccionario con mejores parámetros y resultados
        """
        print(f"\n Búsqueda de hiperparámetros...")
        
        # Grilla por defecto según modelo
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['rbf', 'linear']
                }
        
        # Codificar y escalar
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Grid Search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Actualizar modelo con mejores parámetros
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f" Mejores parámetros encontrados:")
        for param, value in results['best_params'].items():
            print(f"   - {param}: {value}")
        print(f"   - Best CV Score: {results['best_score']:.4f}")
        
        return results
    
    def save_model(self, filename: str = 'models/tactical_classifier.pkl'):
        """
        Guarda el modelo entrenado.
        
        Args:
            filename: Ruta del archivo
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filename)
        print(f" Modelo guardado en: {filename}")
    
    def load_model(self, filename: str):
        """
        Carga un modelo guardado.
        
        Args:
            filename: Ruta del archivo
        """
        model_data = joblib.load(filename)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_fitted = True
        
        print(f" Modelo cargado desde: {filename}")


if __name__ == "__main__":
    # Ejemplo de uso con datos sintéticos
    print(" Probando TacticalStyleClassifier...\n")
    
    # Crear dataset sintético de ejemplo
    np.random.seed(42)
    n_samples = 100
    
    # Features simuladas
    data = {
        'total_passes': np.random.randint(300, 600, n_samples),
        'pass_accuracy': np.random.uniform(0.7, 0.9, n_samples),
        'static_density': np.random.uniform(0.3, 0.7, n_samples),
        'static_avg_clustering': np.random.uniform(0.4, 0.8, n_samples),
        'temporal_density_mean': np.random.uniform(0.3, 0.7, n_samples),
    }
    
    # Labels simuladas (3 estilos: posesión, directo, equilibrado)
    styles = ['posesion', 'directo', 'equilibrado']
    labels = np.random.choice(styles, n_samples)
    
    df = pd.DataFrame(data)
    df['label'] = labels
    
    print(" Dataset sintético creado:")
    print(df.head())
    print(f"\nDistribución de clases: {df['label'].value_counts().to_dict()}")
    
    # Crear clasificador
    classifier = TacticalStyleClassifier(model_type='random_forest')
    
    # Preparar datos
    X, y = classifier.prepare_data(df, target_col='label')
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Entrenar
    classifier.train(X_train, y_train)
    
    # Evaluar
    metrics = classifier.evaluate(X_test, y_test)
    
    # Cross-validation
    cv_results = classifier.cross_validate(X, y, cv=5)
    
    # Feature importance
    print("\n Importancia de features:")
    print(classifier.get_feature_importance())
    
    # Visualizaciones
    classifier.plot_feature_importance(top_n=5)
    classifier.plot_confusion_matrix(X_test, y_test)