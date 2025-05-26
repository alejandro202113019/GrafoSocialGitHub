import pandas as pd
import streamlit as st
from typing import Optional

@st.cache_data
def load_and_process_data(file_path: str = "github_collaboration_data.csv") -> pd.DataFrame:
    """
    Carga y procesa los datos de colaboración de GitHub
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        DataFrame procesado con los datos de colaboración
    """
    try:
        # Cargar el archivo CSV
        df = pd.read_csv(file_path)
        
        # Verificar columnas requeridas
        required_columns = ['developer_source', 'developer_target', 'interaction_type', 'repo', 'weight', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Columnas faltantes en el CSV: {missing_columns}")
        
        # Limpiar y procesar datos
        df = df.dropna(subset=['developer_source', 'developer_target'])
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Validar que no haya self-loops (desarrollador interactuando consigo mismo)
        df = df[df['developer_source'] != df['developer_target']]
        
        # Crear identificadores únicos si es necesario
        df['interaction_id'] = range(len(df))
        
        return df
        
    except FileNotFoundError:
        st.error(f"No se pudo encontrar el archivo: {file_path}")
        st.info("Asegúrate de que el archivo 'github_collaboration_data.csv' esté en el directorio del proyecto")
        raise
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        raise

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Genera un resumen estadístico de los datos
    
    Args:
        df: DataFrame con los datos de colaboración
        
    Returns:
        Diccionario con estadísticas del dataset
    """
    summary = {
        'total_interactions': len(df),
        'unique_developers': len(set(df['developer_source'].unique()) | set(df['developer_target'].unique())),
        'unique_repos': df['repo'].nunique(),
        'interaction_types': df['interaction_type'].value_counts().to_dict(),
        'date_range': {
            'start': df['timestamp'].min() if 'timestamp' in df.columns and df['timestamp'].notna().any() else None,
            'end': df['timestamp'].max() if 'timestamp' in df.columns and df['timestamp'].notna().any() else None
        },
        'weight_stats': {
            'mean': df['weight'].mean(),
            'median': df['weight'].median(),
            'std': df['weight'].std(),
            'min': df['weight'].min(),
            'max': df['weight'].max()
        }
    }
    
    return summary

def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Valida la calidad de los datos y reporta posibles problemas
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        Diccionario con métricas de calidad
    """
    quality_report = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_interactions': df.duplicated(['developer_source', 'developer_target', 'interaction_type', 'repo']).sum(),
        'self_loops': (df['developer_source'] == df['developer_target']).sum(),
        'zero_weights': (df['weight'] == 0).sum(),
        'negative_weights': (df['weight'] < 0).sum()
    }
    
    return quality_report

def create_sample_data() -> pd.DataFrame:
    """
    Crea datos de muestra para testing si no existe el archivo principal
    
    Returns:
        DataFrame con datos de muestra
    """
    import random
    import datetime
    
    # Desarrolladores de muestra
    developers = [
        'alice_dev', 'bob_coder', 'charlie_eng', 'diana_dev', 'eve_programmer',
        'frank_dev', 'grace_coder', 'henry_eng', 'ida_dev', 'jack_programmer',
        'kate_dev', 'liam_coder', 'maya_eng', 'noah_dev', 'olivia_programmer'
    ]
    
    # Repositorios de muestra
    repos = [
        'web-frontend', 'api-backend', 'mobile-app', 'data-pipeline', 
        'auth-service', 'notification-system', 'analytics-dashboard'
    ]
    
    # Tipos de interacción
    interaction_types = ['commit_review', 'pull_request', 'issue_comment']
    
    # Generar datos de muestra
    sample_data = []
    
    for _ in range(200):  # 200 interacciones de muestra
        source = random.choice(developers)
        target = random.choice([d for d in developers if d != source])  # Evitar self-loops
        
        sample_data.append({
            'developer_source': source,
            'developer_target': target,
            'interaction_type': random.choice(interaction_types),
            'repo': random.choice(repos),
            'weight': random.randint(1, 10),
            'timestamp': datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 90))
        })
    
    return pd.DataFrame(sample_data)