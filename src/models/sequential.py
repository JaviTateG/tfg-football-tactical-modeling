"""
Módulo para modelos secuenciales (LSTM y GRU) para análisis táctico de fútbol.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    print(" TensorFlow cargado:", tf.__version__)
except ImportError:
    raise ImportError("TensorFlow no instalado. Ejecuta: pip install tensorflow")


class SequentialTacticalModel:
    """
    Modelos secuenciales (LSTM/GRU) para análisis táctico de fútbol.
    Soporta dos tareas:
      - Clasificación: anticipación de jugadas peligrosas
      - Regresión:     estimación de probabilidad de éxito ofensivo
    """

    def __init__(
        self,
        model_type: str = 'lstm',
        task: str = 'classification',
        sequence_length: int = 10,
        units: int = 64,
        dropout: float = 0.3,
        random_state: int = 42
    ):
        """
        Args:
            model_type: 'lstm' o 'gru'
            task: 'classification' o 'regression'
            sequence_length: Longitud de la ventana temporal
            units: Neuronas en la capa recurrente
            dropout: Tasa de dropout
            random_state: Semilla para reproducibilidad
        """
        self.model_type = model_type
        self.task = task
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.is_fitted = False
        self.n_classes = None
        self.feature_names = None

        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def build_model(self, input_shape: Tuple, n_outputs: int):
        """
        Construye la arquitectura LSTM o GRU.

        Args:
            input_shape: (sequence_length, n_features)
            n_outputs: número de clases o 1 para regresión
        """
        model = Sequential()

        # Capa recurrente principal
        if self.model_type == 'lstm':
            model.add(LSTM(
                self.units,
                input_shape=input_shape,
                return_sequences=True
            ))
        elif self.model_type == 'gru':
            model.add(GRU(
                self.units,
                input_shape=input_shape,
                return_sequences=True
            ))
        else:
            raise ValueError(f"model_type debe ser 'lstm' o 'gru'")

        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        # Segunda capa recurrente
        if self.model_type == 'lstm':
            model.add(LSTM(self.units // 2))
        else:
            model.add(GRU(self.units // 2))

        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        # Capas densas
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout / 2))

        # Capa de salida
        if self.task == 'classification':
            if n_outputs == 2:
                model.add(Dense(1, activation='sigmoid'))
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
            else:
                model.add(Dense(n_outputs, activation='softmax'))
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
        else:  # regression
            model.add(Dense(1, activation='linear'))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )

        self.model = model
        print(f"\n Modelo {self.model_type.upper()} construido:")
        model.summary()

        return model

    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convierte datos tabulares en secuencias para LSTM/GRU.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            X_seq: (n_sequences, sequence_length, n_features)
            y_seq: (n_sequences,)
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Entrena el modelo.

        Args:
            X_train: Secuencias de entrenamiento (n, seq_len, features)
            y_train: Labels de entrenamiento
            X_val: Secuencias de validación (opcional)
            y_val: Labels de validación (opcional)
            epochs: Épocas máximas
            batch_size: Tamaño del batch
        """
        print(f"\n Entrenando {self.model_type.upper()} ({self.task})...")
        print(f"   - Input shape: {X_train.shape}")
        print(f"   - Epochs: {epochs} | Batch: {batch_size}")

        n_features = X_train.shape[2]
        input_shape = (self.sequence_length, n_features)

        # Preparar labels
        if self.task == 'classification':
            if y_train.dtype == object:
                y_train = self.label_encoder.fit_transform(y_train)
                if y_val is not None:
                    y_val = self.label_encoder.transform(y_val)

            self.n_classes = len(np.unique(y_train))

            if self.n_classes > 2:
                y_train_model = to_categorical(y_train, self.n_classes)
                y_val_model = to_categorical(y_val, self.n_classes) if y_val is not None else None
            else:
                y_train_model = y_train
                y_val_model = y_val
        else:
            y_train_model = y_train
            y_val_model = y_val
            self.n_classes = 1

        # Construir modelo
        n_outputs = self.n_classes if self.task == 'classification' else 1
        self.build_model(input_shape, n_outputs)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if y_val is not None else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if y_val is not None else 'loss',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]

        # Datos de validación
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val_model)

        # Entrenar
        self.history = self.model.fit(
            X_train, y_train_model,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        self.is_fitted = True
        print(f"\n Entrenamiento completado en {len(self.history.history['loss'])} épocas")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones.

        Args:
            X: Secuencias (n, seq_len, features)

        Returns:
            Predicciones
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecuta .fit() primero")

        raw = self.model.predict(X, verbose=0)

        if self.task == 'classification':
            if self.n_classes > 2:
                return np.argmax(raw, axis=1)
            else:
                return (raw.flatten() > 0.5).astype(int)
        else:
            return raw.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predice probabilidades (solo clasificación).

        Args:
            X: Secuencias (n, seq_len, features)

        Returns:
            Probabilidades por clase
        """
        if self.task != 'classification':
            raise ValueError("predict_proba solo disponible para clasificación")

        raw = self.model.predict(X, verbose=0)

        if self.n_classes == 2:
            proba = raw.flatten()
            return np.column_stack([1 - proba, proba])
        else:
            return raw

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evalúa el modelo en datos de test.

        Args:
            X_test: Secuencias de test
            y_test: Labels de test

        Returns:
            Diccionario con métricas
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")

        # Codificar si necesario
        if self.task == 'classification' and y_test.dtype == object:
            y_test = self.label_encoder.transform(y_test)

        y_pred = self.predict(X_test)

        metrics = {}

        if self.task == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')

            proba = self.predict_proba(X_test)
            if self.n_classes == 2:
                metrics['auc_roc'] = roc_auc_score(y_test, proba[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_test, classes=range(self.n_classes))
                metrics['auc_roc'] = roc_auc_score(
                    y_bin, proba,
                    average='weighted', multi_class='ovr'
                )

            print(f"\n EVALUACIÓN {self.model_type.upper()}:")
            print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   - F1-Score:  {metrics['f1_score']:.4f}")
            print(f"   - AUC-ROC:   {metrics['auc_roc']:.4f}")
            print("\n Reporte detallado:")
            print(classification_report(y_test, y_pred))

        else:
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['r2'] = r2_score(y_test, y_pred)

            print(f"\n EVALUACIÓN {self.model_type.upper()}:")
            print(f"   - MAE:   {metrics['mae']:.4f}")
            print(f"   - RMSE:  {metrics['rmse']:.4f}")
            print(f"   - R²:    {metrics['r2']:.4f}")

        return metrics

    def plot_training_history(self):
        """
        Visualiza la curva de entrenamiento.
        """
        if self.history is None:
            raise ValueError("Modelo no entrenado")

        metric = 'accuracy' if self.task == 'classification' else 'mae'
        loss = 'loss'

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(self.history.history[loss], label='Train', linewidth=2)
        if f'val_{loss}' in self.history.history:
            axes[0].plot(
                self.history.history[f'val_{loss}'],
                label='Validación', linewidth=2, linestyle='--'
            )
        axes[0].set_title(
            f'Curva de Loss — {self.model_type.upper()}',
            fontsize=13, fontweight='bold'
        )
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Métrica
        if metric in self.history.history:
            axes[1].plot(
                self.history.history[metric], label='Train', linewidth=2
            )
            if f'val_{metric}' in self.history.history:
                axes[1].plot(
                    self.history.history[f'val_{metric}'],
                    label='Validación', linewidth=2, linestyle='--'
                )
            axes[1].set_title(
                f'{metric.upper()} — {self.model_type.upper()}',
                fontsize=13, fontweight='bold'
            )
            axes[1].set_xlabel('Época')
            axes[1].set_ylabel(metric.upper())
            axes[1].legend()
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str):
        """
        Guarda el modelo entrenado.

        Args:
            filepath: Ruta sin extensión (se añade .keras)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")

        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.model.save(f"{filepath}.keras")

        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'task': self.task,
            'sequence_length': self.sequence_length,
            'n_classes': self.n_classes,
            'feature_names': self.feature_names
        }, f"{filepath}_metadata.pkl")

        print(f" Modelo guardado en: {filepath}.keras")

    def load_model(self, filepath: str):
        """
        Carga un modelo guardado.

        Args:
            filepath: Ruta sin extensión
        """
        import joblib

        self.model = tf.keras.models.load_model(f"{filepath}.keras")

        metadata = joblib.load(f"{filepath}_metadata.pkl")
        self.scaler = metadata['scaler']
        self.label_encoder = metadata['label_encoder']
        self.model_type = metadata['model_type']
        self.task = metadata['task']
        self.sequence_length = metadata['sequence_length']
        self.n_classes = metadata['n_classes']
        self.feature_names = metadata['feature_names']
        self.is_fitted = True

        print(f" Modelo cargado desde: {filepath}.keras")