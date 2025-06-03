import streamlit as st
import pandas as pd
import networkx as nx

# Configuración básica
st.set_page_config(page_title="Test App", layout="wide")

st.title("🧪 Test de Carga de Módulos")

# Test 1: Verificar pandas y networkx
try:
    df_test = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    G_test = nx.Graph()
    G_test.add_edges_from([(1, 2), (2, 3)])
    st.success("✅ pandas y networkx funcionan correctamente")
    st.write("DataFrame test:", df_test)
    st.write("Graph test nodes:", list(G_test.nodes()))
except Exception as e:
    st.error(f"❌ Error con pandas/networkx: {e}")

# Test 2: Verificar data_loader
st.subheader("Test data_loader")
try:
    from data_loader import load_and_process_data
    st.success("✅ data_loader importado correctamente")
    
    # Intentar cargar datos
    df = load_and_process_data()
    st.success(f"✅ Datos cargados: {len(df)} filas")
    st.dataframe(df.head())
    
except Exception as e:
    st.error(f"❌ Error con data_loader: {e}")

# Test 3: Verificar network_analyzer
st.subheader("Test network_analyzer")
try:
    from network_analyzer import NetworkAnalyzer
    st.success("✅ network_analyzer importado correctamente")
except Exception as e:
    st.error(f"❌ Error con network_analyzer: {e}")

# Test 4: Verificar ai_optimizer
st.subheader("Test ai_optimizer")
try:
    from ai_optimizer import AINetworkOptimizer
    st.success("✅ ai_optimizer importado correctamente")
except Exception as e:
    st.error(f"❌ Error con ai_optimizer: {e}")

# Test 5: Verificar visualizer
st.subheader("Test visualizer")
try:
    from visualizer import NetworkVisualizer
    st.success("✅ visualizer importado correctamente")
except Exception as e:
    st.error(f"❌ Error con visualizer: {e}")

# Test 6: Verificar community_detector
st.subheader("Test community_detector")
try:
    from community_detector import AIOptimizedCommunityDetector
    st.success("✅ community_detector importado correctamente")
except Exception as e:
    st.error(f"❌ Error con community_detector: {e}")

# Test 7: Verificar scikit-learn
st.subheader("Test scikit-learn")
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    st.success("✅ scikit-learn disponible")
except Exception as e:
    st.warning(f"⚠️ scikit-learn no disponible: {e}")
    st.info("Instala con: pip install scikit-learn")

# Test 8: Verificar archivo CSV
st.subheader("Test archivo CSV")
try:
    import os
    if os.path.exists("github_collaboration_data.csv"):
        st.success("✅ Archivo CSV encontrado")
        df_csv = pd.read_csv("github_collaboration_data.csv")
        st.write(f"Filas: {len(df_csv)}, Columnas: {list(df_csv.columns)}")
    else:
        st.error("❌ Archivo github_collaboration_data.csv no encontrado")
except Exception as e:
    st.error(f"❌ Error leyendo CSV: {e}")

st.subheader("📋 Resumen")
st.info("Si todos los tests pasan ✅, la aplicación principal debería funcionar correctamente.")