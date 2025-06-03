import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Importar módulos locales
from data_loader import load_and_process_data
from network_analyzer import NetworkAnalyzer
from visualizer import NetworkVisualizer
from community_detector import AIOptimizedCommunityDetector
from ai_optimizer import AINetworkOptimizer

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Redes Sociales con IA - GitHub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2E86AB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .ai-highlight {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .stDataFrame {
        background-color: white;
    }
    .optimization-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .comparison-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Título principal con énfasis en IA
    st.markdown('<h1 class="main-header">🤖 Análisis de Redes Sociales con IA en GitHub</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="ai-highlight">✨ Potenciado por Algoritmos de Inteligencia Artificial para Optimización de Colaboraciones</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=100)
        st.title("🤖 Panel de Control IA")
        st.markdown("---")
        
        # Selector de sección
        section = st.selectbox(
            "Selecciona una sección:",
            [
                "📈 Resumen General", 
                "🔍 Análisis de Red Base", 
                "👥 Comunidades IA", 
                "🏆 Líderes Técnicos",
                "🤖 Optimización IA Completa",
                "📊 Patrones Colaborativos",
                "📋 Recomendaciones IA",
                "🔬 Análisis Comparativo Antes/Después"
            ]
        )
        
        st.markdown("---")
        
        # Configuraciones de IA
        st.markdown("### ⚙️ Configuración IA")
        
        community_method = st.selectbox(
            "Método de Detección de Comunidades:",
            ['hybrid_ai', 'spectral_ai', 'kmeans_ai', 'greedy', 'louvain']
        )
        
        optimization_level = st.slider(
            "Nivel de Optimización IA:",
            min_value=1, max_value=5, value=3,
            help="1=Básico, 5=Máximo (más lento)"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ Información")
        st.info("Este dashboard utiliza algoritmos avanzados de IA para analizar y optimizar colaboraciones en GitHub.")
    
    # Cargar datos
    try:
        with st.spinner("🔄 Cargando y procesando datos..."):
            df = load_and_process_data()
            G = create_network_graph(df)
            
            # Inicializar componentes con IA
            analyzer = NetworkAnalyzer(G)
            visualizer = NetworkVisualizer(G)
            ai_community_detector = AIOptimizedCommunityDetector(G)
            ai_optimizer = AINetworkOptimizer(G, df)
            
            # Calcular métricas básicas
            metrics = analyzer.calculate_all_metrics()
        
        st.success("✅ Datos cargados y analizados exitosamente")
        
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")
        st.info("💡 Asegúrate de tener instaladas todas las dependencias: `pip install scikit-learn`")
        st.stop()
    
    # Mostrar sección seleccionada
    if section == "📈 Resumen General":
        show_general_overview(df, G, metrics)
    elif section == "🔍 Análisis de Red Base":
        show_base_network_analysis(G, metrics, visualizer, df)
    elif section == "👥 Comunidades IA":
        show_ai_community_analysis(G, ai_community_detector, df, community_method)
    elif section == "🏆 Líderes Técnicos":
        show_technical_leaders_classification(metrics, G, df, ai_optimizer)
    elif section == "🤖 Optimización IA Completa":
        show_complete_ai_optimization(ai_optimizer, G, df, visualizer)
    elif section == "📊 Patrones Colaborativos":
        show_collaboration_patterns(ai_optimizer)
    elif section == "📋 Recomendaciones IA":
        show_ai_recommendations(G, metrics, ai_optimizer, df)
    elif section == "🔬 Análisis Comparativo Antes/Después":
        show_before_after_analysis(G, df, ai_optimizer, visualizer)

def create_network_graph(df):
    """Crear grafo de NetworkX desde el DataFrame"""
    G = nx.DiGraph()
    
    # Agregar nodos
    developers = set(df['developer_source'].unique()) | set(df['developer_target'].unique())
    G.add_nodes_from(developers)
    
    # Agregar aristas con pesos
    for _, row in df.iterrows():
        source = row['developer_source']
        target = row['developer_target']
        weight = row['weight']
        
        if G.has_edge(source, target):
            G[source][target]['weight'] += weight
        else:
            G.add_edge(source, target, weight=weight)
    
    return G

def calculate_detailed_metrics(G):
    """Calcula métricas detalladas del grafo"""
    metrics = {}
    
    # Métricas básicas
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Métricas de centralidad
    metrics['pagerank'] = nx.pagerank(G, weight='weight')
    metrics['betweenness_centrality'] = nx.betweenness_centrality(G, weight='weight')
    metrics['degree_centrality'] = nx.degree_centrality(G)
    metrics['closeness_centrality'] = nx.closeness_centrality(G, distance='weight')
    
    try:
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except:
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Métricas globales
    undirected_G = G.to_undirected()
    metrics['avg_clustering'] = nx.average_clustering(undirected_G, weight='weight')
    metrics['num_components'] = nx.number_connected_components(undirected_G)
    metrics['reciprocity'] = nx.reciprocity(G)
    
    if nx.is_connected(undirected_G):
        metrics['diameter'] = nx.diameter(undirected_G)
        metrics['avg_path_length'] = nx.average_shortest_path_length(undirected_G)
    else:
        metrics['diameter'] = float('inf')
        metrics['avg_path_length'] = float('inf')
    
    return metrics

def show_base_network_analysis(G, metrics, visualizer, df):
    """Análisis de la Red de Colaboración Base según el taller"""
    st.markdown('<h2 class="section-header">🔍 Análisis de la Red de Colaboración Base</h2>', unsafe_allow_html=True)
    
    # Métricas estructurales de la red base (Tabla del taller)
    st.subheader("📊 Métricas Estructurales de la Red Base")
    
    # Calcular métricas específicas como en el taller
    undirected_G = G.to_undirected()
    
    base_metrics = {
        'Densidad de red': nx.density(G),
        'Clustering promedio': nx.average_clustering(undirected_G, weight='weight'),
        'Componentes conectados': nx.number_connected_components(undirected_G),
        'Diámetro de red': nx.diameter(undirected_G) if nx.is_connected(undirected_G) else float('inf'),
        'Reciprocidad': nx.reciprocity(G)
    }
    
    # Crear tabla como en el taller
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Densidad de red", f"{base_metrics['Densidad de red']:.3f}")
        st.write("*Red moderadamente conectada*")
    
    with col2:
        st.metric("🕸️ Clustering promedio", f"{base_metrics['Clustering promedio']:.3f}")
        st.write("*Alta tendencia a formación de grupos*")
    
    with col3:
        st.metric("🔗 Componentes conectados", base_metrics['Componentes conectados'])
        st.write("*Red completamente conectada*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        diameter_val = base_metrics['Diámetro de red']
        if diameter_val != float('inf'):
            st.metric("📏 Diámetro de red", diameter_val)
            st.write("*Comunicación eficiente*")
        else:
            st.metric("📏 Diámetro de red", "∞")
            st.write("*Red desconectada*")
    
    with col2:
        st.metric("🔄 Reciprocidad", f"{base_metrics['Reciprocidad']:.3f}")
        st.write("*Alta colaboración bidireccional*")
    
    # Tabla de métricas estructurales como en el documento
    st.subheader("📋 Cuadro 1: Métricas estructurales de la red base")
    
    metrics_df = pd.DataFrame([
        {'Métrica': 'Densidad de red', 'Valor Inicial': f"{base_metrics['Densidad de red']:.3f}", 'Interpretación': 'Red moderadamente conectada'},
        {'Métrica': 'Clustering promedio', 'Valor Inicial': f"{base_metrics['Clustering promedio']:.3f}", 'Interpretación': 'Alta tendencia a formación de grupos'},
        {'Métrica': 'Componentes conectados', 'Valor Inicial': base_metrics['Componentes conectados'], 'Interpretación': 'Red completamente conectada'},
        {'Métrica': 'Diámetro de red', 'Valor Inicial': base_metrics['Diámetro de red'] if base_metrics['Diámetro de red'] != float('inf') else '∞', 'Interpretación': 'Comunicación eficiente'},
        {'Métrica': 'Reciprocidad', 'Valor Inicial': f"{base_metrics['Reciprocidad']:.3f}", 'Interpretación': 'Alta colaboración bidireccional'}
    ])
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Información del dataset como en el taller
    st.subheader("📈 Información del Dataset")
    
    developers = set(df['developer_source'].unique()) | set(df['developer_target'].unique())
    repos = df['repo'].unique()
    interactions = len(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Desarrolladores", len(developers))
    
    with col2:
        st.metric("📁 Repositorios principales", len(repos))
    
    with col3:
        st.metric("🔄 Interacciones documentadas", interactions)
    
    with col4:
        # Calcular período (aproximado)
        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            date_range = pd.to_datetime(df['timestamp']).max() - pd.to_datetime(df['timestamp']).min()
            period = f"{date_range.days} días"
        else:
            period = "3 meses (estimado)"
        st.metric("📅 Período", period)
    
    # Visualización de la red base
    st.subheader("🕸️ Visualización de la Red Base")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        metric_option = st.selectbox(
            "Métrica para colorear:",
            ['pagerank', 'betweenness', 'closeness', 'eigenvector'],
            key="base_metric"
        )
    
    with col1:
        fig = visualizer.create_network_plot(metrics, metric_option, 
                                           title="Red de Colaboración Base - GitHub")
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis detallado por repositorio
    st.subheader("📁 Análisis por Repositorio")
    
    repo_analysis = []
    for repo in repos:
        repo_data = df[df['repo'] == repo]
        repo_devs = set(repo_data['developer_source']) | set(repo_data['developer_target'])
        
        repo_analysis.append({
            'Repositorio': repo,
            'Interacciones': len(repo_data),
            'Desarrolladores': len(repo_devs),
            'Peso Promedio': repo_data['weight'].mean(),
            'Tipos de Interacción': len(repo_data['interaction_type'].unique())
        })
    
    repo_df = pd.DataFrame(repo_analysis)
    st.dataframe(repo_df.style.format({
        'Peso Promedio': '{:.2f}'
    }).background_gradient(subset=['Interacciones']), use_container_width=True)

def show_technical_leaders_classification(metrics, G, df, ai_optimizer):
    """Sistema de Clasificación Inteligente de Desarrolladores como en el taller"""
    st.markdown('<h2 class="section-header">🏆 Sistema de Clasificación Inteligente de Desarrolladores</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="ai-highlight">🤖 Algoritmo de scoring multidimensional para clasificación automática</div>', unsafe_allow_html=True)
    
    # Calcular scoring multidimensional como en el taller
    developers = list(G.nodes())
    
    # Pesos para el scoring (como en el taller)
    weights = {
        'pagerank': 0.35,
        'betweenness': 0.30,
        'eigenvector': 0.20,
        'closeness': 0.15
    }
    
    # Calcular scores IA
    ai_scores = {}
    for dev in developers:
        score = 0
        score += metrics['pagerank'][dev] * weights['pagerank'] * 10  # Normalizar
        score += metrics['betweenness'][dev] * weights['betweenness'] * 10
        score += metrics['eigenvector'][dev] * weights['eigenvector'] * 10
        score += metrics['closeness'][dev] * weights['closeness'] * 10
        ai_scores[dev] = score
    
    # Clasificar en roles (como en el taller)
    sorted_devs = sorted(ai_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Determinar especialización basada en datos
    specializations = {}
    for dev in developers:
        dev_data = df[(df['developer_source'] == dev) | (df['developer_target'] == dev)]
        if len(dev_data) > 0:
            # Determinar especialización por repositorio más frecuente
            top_repo = dev_data['repo'].value_counts().index[0] if len(dev_data) > 0 else 'general'
            if 'frontend' in top_repo.lower():
                specializations[dev] = 'Frontend/Interfaces'
            elif 'backend' in top_repo.lower() or 'api' in top_repo.lower():
                specializations[dev] = 'Backend/Servicios'
            elif 'data' in top_repo.lower():
                specializations[dev] = 'Datos/Analytics'
            else:
                specializations[dev] = 'Desarrollo General'
        else:
            specializations[dev] = 'Sin especialización'
    
    # Asignar roles basado en percentiles (como en el taller)
    scores = list(ai_scores.values())
    threshold_leader = np.percentile(scores, 80)
    threshold_connector = np.percentile(scores, 60)
    threshold_senior = np.percentile(scores, 40)
    
    classified_devs = []
    for dev, score in sorted_devs:
        if score >= threshold_leader and metrics['betweenness'][dev] > np.percentile(list(metrics['betweenness'].values()), 70):
            role = 'Líder Técnico'
        elif metrics['betweenness'][dev] > np.percentile(list(metrics['betweenness'].values()), 80):
            role = 'Conector Principal'
        elif score >= threshold_senior:
            role = 'Colaborador Senior'
        elif 'backend' in specializations[dev].lower() or 'data' in specializations[dev].lower():
            role = 'Especialista Backend'
        elif 'frontend' in specializations[dev].lower():
            role = 'Desarrollador Frontend'
        else:
            role = 'Desarrollador Junior'
        
        classified_devs.append({
            'Desarrollador': dev,
            'Rol Identificado': role,
            'Score IA': f"{score:.3f}",
            'Especialización': specializations[dev],
            'PageRank': f"{metrics['pagerank'][dev]:.3f}",
            'Intermediación': f"{metrics['betweenness'][dev]:.3f}",
            'Colaboraciones': len(df[(df['developer_source'] == dev) | (df['developer_target'] == dev)])
        })
    
    # Mostrar tabla de clasificación como en el taller (Cuadro 2)
    st.subheader("📋 Cuadro 2: Clasificación automática de roles por IA")
    
    classification_df = pd.DataFrame(classified_devs)
    
    st.dataframe(
        classification_df.style.background_gradient(subset=['Score IA']),
        use_container_width=True
    )
    
    # Análisis de distribución de roles
    st.subheader("📊 Distribución de Roles Identificados")
    
    role_counts = classification_df['Rol Identificado'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(values=role_counts.values, names=role_counts.index,
                    title="Distribución de Roles por IA")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(x=role_counts.index, y=role_counts.values,
                    title="Cantidad por Rol")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart para top 5 desarrolladores
    st.subheader("📡 Perfil Multidimensional - Top 5 Desarrolladores")
    
    top_5 = classified_devs[:5]
    
    categories = ['PageRank', 'Intermediación', 'Eigenvector', 'Cercanía', 'Colaboraciones']
    
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, dev_data in enumerate(top_5):
        dev = dev_data['Desarrollador']
        collab_count = dev_data['Colaboraciones']
        max_collabs = max([d['Colaboraciones'] for d in classified_devs])
        
        values = [
            metrics['pagerank'][dev] / max(metrics['pagerank'].values()),
            metrics['betweenness'][dev] / max(metrics['betweenness'].values()) if max(metrics['betweenness'].values()) > 0 else 0,
            metrics['eigenvector'][dev] / max(metrics['eigenvector'].values()) if max(metrics['eigenvector'].values()) > 0 else 0,
            metrics['closeness'][dev] / max(metrics['closeness'].values()),
            collab_count / max_collabs
        ]
        values += values[:1]  # Cerrar el polígono
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=dev,
            line_color=colors[i],
            fillcolor=colors[i],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title="Análisis Multidimensional de Líderes Técnicos",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_complete_ai_optimization(ai_optimizer, G, df, visualizer):
    """Optimización completa con métricas antes/después"""
    st.markdown('<h2 class="section-header">🤖 Optimización Completa con IA</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="optimization-card">🚀 Análisis completo: Antes → Algoritmo de Optimización → Después</div>', unsafe_allow_html=True)
    
    # Métricas antes de la optimización
    st.subheader("📊 Estado Actual (Antes de Optimización)")
    
    original_metrics = calculate_detailed_metrics(G)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🏘️ Nodos", original_metrics['num_nodes'])
    with col2:
        st.metric("🔗 Aristas", original_metrics['num_edges'])
    with col3:
        st.metric("📊 Densidad", f"{original_metrics['density']:.3f}")
    with col4:
        st.metric("🕸️ Clustering", f"{original_metrics['avg_clustering']:.3f}")
    with col5:
        st.metric("🔄 Reciprocidad", f"{original_metrics['reciprocity']:.3f}")
    
    # Top nodos por centralidad de intermediación (antes)
    st.subheader("🌉 Top Nodos - Centralidad de Intermediación (ANTES)")
    
    betweenness_before = sorted(original_metrics['betweenness_centrality'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
    
    bet_df_before = pd.DataFrame(betweenness_before, columns=['Desarrollador', 'Intermediación'])
    bet_df_before['Rango'] = range(1, len(bet_df_before) + 1)
    
    fig_bet_before = px.bar(bet_df_before, x='Intermediación', y='Desarrollador',
                           orientation='h', title="Centralidad de Intermediación - ANTES",
                           color='Intermediación', color_continuous_scale='viridis')
    st.plotly_chart(fig_bet_before, use_container_width=True)
    
    # Top nodos por centralidad de grado (antes)
    st.subheader("📊 Top Nodos - Centralidad de Grado (ANTES)")
    
    degree_before = sorted(original_metrics['degree_centrality'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
    
    deg_df_before = pd.DataFrame(degree_before, columns=['Desarrollador', 'Grado'])
    deg_df_before['Rango'] = range(1, len(deg_df_before) + 1)
    
    fig_deg_before = px.bar(deg_df_before, x='Grado', y='Desarrollador',
                           orientation='h', title="Centralidad de Grado - ANTES",
                           color='Grado', color_continuous_scale='plasma')
    st.plotly_chart(fig_deg_before, use_container_width=True)
    
    # Ejecutar optimización
    st.subheader("🚀 Ejecutar Optimización IA")
    
    if st.button("🤖 Ejecutar Optimización Completa del Grafo", type="primary"):
        with st.spinner("🔄 Ejecutando algoritmos de optimización..."):
            
            # Aplicar optimización usando el optimizador mejorado
            G_optimized = ai_optimizer.apply_optimization_recommendations(top_recommendations=5)
            
            # Obtener comparación completa
            comparison_results = ai_optimizer.get_optimization_comparison()
            
            # Guardar en session state para mantener resultados
            st.session_state['original_metrics'] = original_metrics
            st.session_state['optimized_metrics'] = ai_optimizer.optimized_metrics
            st.session_state['G_optimized'] = G_optimized
            st.session_state['optimization_applied'] = True
            st.session_state['comparison_results'] = comparison_results
        
        st.success("✅ Optimización completada!")
        st.rerun()
    
    # Mostrar resultados si ya se ejecutó la optimización
    if st.session_state.get('optimization_applied', False):
        
        optimized_metrics = st.session_state['optimized_metrics']
        G_optimized = st.session_state['G_optimized']
        comparison_results = st.session_state.get('comparison_results', {})
        
        st.subheader("📈 Estado Después de Optimización")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            delta_nodes = optimized_metrics['num_nodes'] - original_metrics['num_nodes']
            st.metric("🏘️ Nodos", optimized_metrics['num_nodes'], delta=delta_nodes)
        with col2:
            delta_edges = optimized_metrics['num_edges'] - original_metrics['num_edges']
            st.metric("🔗 Aristas", optimized_metrics['num_edges'], delta=delta_edges)
        with col3:
            delta_density = optimized_metrics['density'] - original_metrics['density']
            st.metric("📊 Densidad", f"{optimized_metrics['density']:.3f}", 
                     delta=f"{delta_density:+.3f}")
        with col4:
            delta_clustering = optimized_metrics['avg_clustering'] - original_metrics['avg_clustering']
            st.metric("🕸️ Clustering", f"{optimized_metrics['avg_clustering']:.3f}", 
                     delta=f"{delta_clustering:+.3f}")
        with col5:
            delta_reciprocity = optimized_metrics['reciprocity'] - original_metrics['reciprocity']
            st.metric("🔄 Reciprocidad", f"{optimized_metrics['reciprocity']:.3f}", 
                     delta=f"{delta_reciprocity:+.3f}")
        
        # Grafo después de la optimización
        st.subheader("🕸️ Grafo Después de Aplicar el Algoritmo de Optimización")
        
        # Crear visualizador para el grafo optimizado
        visualizer_optimized = NetworkVisualizer(G_optimized)
        metrics_optimized = {
            'pagerank': optimized_metrics['pagerank'],
            'betweenness': optimized_metrics['betweenness_centrality'],
            'closeness': optimized_metrics['closeness_centrality'],
            'eigenvector': optimized_metrics['eigenvector_centrality']
        }
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            metric_option_opt = st.selectbox(
                "Métrica para colorear (optimizado):",
                ['pagerank', 'betweenness', 'closeness', 'eigenvector'],
                key="optimized_metric"
            )
        
        with col1:
            fig_optimized = visualizer_optimized.create_network_plot(
                metrics_optimized, metric_option_opt, 
                title="Grafo OPTIMIZADO - Red de Colaboración"
            )
            st.plotly_chart(fig_optimized, use_container_width=True)
        
        # Top nodos por centralidad de intermediación (después)
        st.subheader("🌉 Top Nodos - Centralidad de Intermediación (DESPUÉS)")
        
        betweenness_after = sorted(optimized_metrics['betweenness_centrality'].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
        
        bet_df_after = pd.DataFrame(betweenness_after, columns=['Desarrollador', 'Intermediación'])
        bet_df_after['Rango'] = range(1, len(bet_df_after) + 1)
        
        fig_bet_after = px.bar(bet_df_after, x='Intermediación', y='Desarrollador',
                              orientation='h', title="Centralidad de Intermediación - DESPUÉS",
                              color='Intermediación', color_continuous_scale='viridis')
        st.plotly_chart(fig_bet_after, use_container_width=True)
        
        # Top nodos por centralidad de grado (después)
        st.subheader("📊 Top Nodos - Centralidad de Grado (DESPUÉS)")
        
        degree_after = sorted(optimized_metrics['degree_centrality'].items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        
        deg_df_after = pd.DataFrame(degree_after, columns=['Desarrollador', 'Grado'])
        deg_df_after['Rango'] = range(1, len(deg_df_after) + 1)
        
        fig_deg_after = px.bar(deg_df_after, x='Grado', y='Desarrollador',
                              orientation='h', title="Centralidad de Grado - DESPUÉS",
                              color='Grado', color_continuous_scale='plasma')
        st.plotly_chart(fig_deg_after, use_container_width=True)
        
        # Resumen de mejoras conseguidas
        if comparison_results.get('improvement_summary'):
            improvement_summary = comparison_results['improvement_summary']
            
            st.subheader("🎯 Resumen de Mejoras Conseguidas")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📈 Métricas Mejoradas",
                    f"{improvement_summary['metrics_improved']}/{improvement_summary['total_metrics']}",
                    help="Número de métricas que mejoraron"
                )
            
            with col2:
                st.metric(
                    "🔗 Nuevas Conexiones",
                    improvement_summary['new_connections'],
                    help="Conexiones completamente nuevas añadidas"
                )
            
            with col3:
                st.metric(
                    "💪 Conexiones Reforzadas",
                    improvement_summary['reinforced_connections'],
                    help="Conexiones existentes que se reforzaron"
                )
            
            with col4:
                st.metric(
                    "🚧 Cuellos de Botella Mitigados",
                    improvement_summary['bottlenecks_mitigated'],
                    help="Cuellos de botella críticos que se aliviaron"
                )
        
        # Tabla comparativa detallada de métricas globales
        st.subheader("📋 Comparación Detallada: Métricas Globales")
        
        if comparison_results.get('metrics_comparison'):
            comparison_data = []
            
            for metric_name, metric_data in comparison_results['metrics_comparison'].items():
                comparison_data.append({
                    'Métrica': metric_name.replace('_', ' ').title(),
                    'Antes': f"{metric_data['before']:.4f}" if isinstance(metric_data['before'], float) else str(metric_data['before']),
                    'Después': f"{metric_data['after']:.4f}" if isinstance(metric_data['after'], float) else str(metric_data['after']),
                    'Cambio': f"{metric_data['change']:+.4f}" if isinstance(metric_data['change'], float) else f"{metric_data['change']:+d}",
                    'Cambio %': f"{metric_data['change_percentage']:+.2f}%" if metric_data['change_percentage'] != 0 else "0.00%",
                    'Mejoró': "✅" if metric_data.get('improved', False) else "➖"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            st.dataframe(
                comparison_df.style.applymap(
                    lambda x: 'background-color: lightgreen' if '✅' in str(x) else 
                             'background-color: lightcoral' if '➖' in str(x) else '',
                    subset=['Mejoró']
                ).applymap(
                    lambda x: 'background-color: lightgreen' if '+' in str(x) and '%' in str(x) and x != '+0.00%' else 
                             'background-color: lightcoral' if '-' in str(x) and '%' in str(x) else '',
                    subset=['Cambio %']
                ),
                use_container_width=True
            )

def show_before_after_analysis(G, df, ai_optimizer, visualizer):
    """Análisis detallado antes/después de optimización"""
    st.markdown('<h2 class="section-header">🔬 Análisis Comparativo Detallado: Antes/Después</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('optimization_applied', False):
        st.info("💡 Para ver la comparación detallada, primero ejecuta la optimización en la sección '🤖 Optimización IA Completa'")
        
        # NUEVA SECCIÓN: Mostrar métricas actuales mientras tanto
        st.subheader("📊 Métricas Actuales de la Red")
        current_metrics = calculate_detailed_metrics(G)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🏘️ Nodos", current_metrics['num_nodes'])
        with col2:
            st.metric("🔗 Aristas", current_metrics['num_edges'])
        with col3:
            st.metric("📊 Densidad", f"{current_metrics['density']:.3f}")
        with col4:
            st.metric("🕸️ Clustering", f"{current_metrics['avg_clustering']:.3f}")
        with col5:
            st.metric("🔄 Reciprocidad", f"{current_metrics['reciprocity']:.3f}")
        
        # Mostrar tabla de centralidad actual
        st.subheader("🏆 Rankings Actuales de Centralidad")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 - PageRank:**")
            pagerank_current = sorted(current_metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:10]
            pagerank_df = pd.DataFrame(pagerank_current, columns=['Desarrollador', 'PageRank'])
            pagerank_df['Rango'] = range(1, len(pagerank_df) + 1)
            st.dataframe(pagerank_df[['Rango', 'Desarrollador', 'PageRank']], use_container_width=True)
        
        with col2:
            st.write("**Top 10 - Intermediación:**")
            betweenness_current = sorted(current_metrics['betweenness_centrality'].items(), key=lambda x: x[1], reverse=True)[:10]
            betweenness_df = pd.DataFrame(betweenness_current, columns=['Desarrollador', 'Intermediación'])
            betweenness_df['Rango'] = range(1, len(betweenness_df) + 1)
            st.dataframe(betweenness_df[['Rango', 'Desarrollador', 'Intermediación']], use_container_width=True)
        
        return
    
    original_metrics = st.session_state['original_metrics']
    optimized_metrics = st.session_state['optimized_metrics']
    G_optimized = st.session_state['G_optimized']
    
    # Comparación visual lado a lado
    st.subheader("👁️ Comparación Visual: Grafos Antes vs Después")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 ANTES de la Optimización")
        
        metrics_before = {
            'pagerank': original_metrics['pagerank'],
            'betweenness': original_metrics['betweenness_centrality'],
            'closeness': original_metrics['closeness_centrality'],
            'eigenvector': original_metrics['eigenvector_centrality']
        }
        
        fig_before = visualizer.create_network_plot(
            metrics_before, 'pagerank', 
            title="Red Original"
        )
        st.plotly_chart(fig_before, use_container_width=True)
        
        # Métricas clave antes
        st.write("**Métricas Clave:**")
        st.write(f"• Densidad: {original_metrics['density']:.3f}")
        st.write(f"• Clustering: {original_metrics['avg_clustering']:.3f}")
        st.write(f"• Aristas: {original_metrics['num_edges']}")
        st.write(f"• Reciprocidad: {original_metrics['reciprocity']:.3f}")
    
    with col2:
        st.markdown("#### 🚀 DESPUÉS de la Optimización")
        
        visualizer_optimized = NetworkVisualizer(G_optimized)
        metrics_after = {
            'pagerank': optimized_metrics['pagerank'],
            'betweenness': optimized_metrics['betweenness_centrality'],
            'closeness': optimized_metrics['closeness_centrality'],
            'eigenvector': optimized_metrics['eigenvector_centrality']
        }
        
        fig_after = visualizer_optimized.create_network_plot(
            metrics_after, 'pagerank', 
            title="Red Optimizada"
        )
        st.plotly_chart(fig_after, use_container_width=True)
        
        # Métricas clave después
        st.write("**Métricas Clave:**")
        st.write(f"• Densidad: {optimized_metrics['density']:.3f}")
        st.write(f"• Clustering: {optimized_metrics['avg_clustering']:.3f}")
        st.write(f"• Aristas: {optimized_metrics['num_edges']}")
        st.write(f"• Reciprocidad: {optimized_metrics['reciprocity']:.3f}")
    
    # Análisis de cambios en ranking
    st.subheader("📈 Cambios en Rankings de Centralidad")
    
    # Betweenness centrality changes
    betweenness_before_rank = {dev: rank for rank, (dev, _) in enumerate(
        sorted(original_metrics['betweenness_centrality'].items(), key=lambda x: x[1], reverse=True), 1)}
    betweenness_after_rank = {dev: rank for rank, (dev, _) in enumerate(
        sorted(optimized_metrics['betweenness_centrality'].items(), key=lambda x: x[1], reverse=True), 1)}
    
    ranking_changes = []
    for dev in G.nodes():
        before_rank = betweenness_before_rank.get(dev, len(G.nodes()))
        after_rank = betweenness_after_rank.get(dev, len(G.nodes()))
        change = before_rank - after_rank  # Positivo = mejora en ranking
        
        ranking_changes.append({
            'Desarrollador': dev,
            'Ranking Antes': before_rank,
            'Ranking Después': after_rank,
            'Cambio': change,
            'Centralidad Antes': f"{original_metrics['betweenness_centrality'][dev]:.3f}",
            'Centralidad Después': f"{optimized_metrics['betweenness_centrality'][dev]:.3f}"
        })
    
    ranking_df = pd.DataFrame(ranking_changes)
    ranking_df = ranking_df.sort_values('Cambio', ascending=False)
    
    st.subheader("🔄 Cambios en Ranking de Centralidad de Intermediación")
    st.dataframe(
        ranking_df.style.applymap(
            lambda x: 'background-color: lightgreen' if isinstance(x, (int, float)) and x > 0 else 
                     'background-color: lightcoral' if isinstance(x, (int, float)) and x < 0 else '',
            subset=['Cambio']
        ),
        use_container_width=True
    )
    
    # Métricas de impacto de la optimización
    st.subheader("🎯 Impacto de la Optimización")
    
    # Calcular mejoras porcentuales
    density_improvement = ((optimized_metrics['density'] - original_metrics['density']) / original_metrics['density']) * 100
    clustering_improvement = ((optimized_metrics['avg_clustering'] - original_metrics['avg_clustering']) / original_metrics['avg_clustering']) * 100
    edges_added = optimized_metrics['num_edges'] - original_metrics['num_edges']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Mejora en Densidad", 
            f"{density_improvement:+.1f}%",
            help="Incremento porcentual en la densidad de la red"
        )
    
    with col2:
        st.metric(
            "🕸️ Mejora en Clustering", 
            f"{clustering_improvement:+.1f}%",
            help="Incremento porcentual en clustering"
        )
    
    with col3:
        st.metric(
            "🔗 Conexiones Añadidas", 
            f"+{edges_added}",
            help="Nuevas conexiones creadas por la optimización"
        )
    
    with col4:
        # Calcular diversidad de conexiones
        original_connections = set()
        for u, v in G.edges():
            original_connections.add((min(u, v), max(u, v)))
        
        optimized_connections = set()
        for u, v in G_optimized.edges():
            optimized_connections.add((min(u, v), max(u, v)))
        
        new_connections = optimized_connections - original_connections
        diversity_improvement = len(new_connections)
        
        st.metric(
            "🎯 Nuevas Colaboraciones", 
            diversity_improvement,
            help="Número de nuevas colaboraciones únicas"
        )
    
    # Análisis de impacto por desarrollador
    st.subheader("👥 Impacto por Desarrollador")
    
    developer_impact = []
    for dev in G.nodes():
        original_degree = G.degree(dev, weight='weight')
        optimized_degree = G_optimized.degree(dev, weight='weight')
        degree_change = optimized_degree - original_degree
        
        original_betweenness = original_metrics['betweenness_centrality'][dev]
        optimized_betweenness = optimized_metrics['betweenness_centrality'][dev]
        betweenness_change = optimized_betweenness - original_betweenness
        
        developer_impact.append({
            'Desarrollador': dev,
            'Cambio en Grado': f"{degree_change:+.1f}",
            'Cambio en Intermediación': f"{betweenness_change:+.3f}",
            'Impacto Total': abs(degree_change) + abs(betweenness_change * 10)  # Score compuesto
        })
    
    impact_df = pd.DataFrame(developer_impact)
    impact_df = impact_df.sort_values('Impacto Total', ascending=False)
    
    st.dataframe(
        impact_df.style.background_gradient(subset=['Impacto Total']),
        use_container_width=True
    )
def show_general_overview(df, G, metrics):
    """Mostrar resumen general del análisis"""
    st.markdown('<h2 class="section-header">📈 Resumen General</h2>', unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Desarrolladores", G.number_of_nodes())
    
    with col2:
        st.metric("🔗 Interacciones", G.number_of_edges())
    
    with col3:
        total_weight = sum([data['weight'] for _, _, data in G.edges(data=True)])
        st.metric("⚖️ Peso Total", total_weight)
    
    with col4:
        density = nx.density(G)
        st.metric("📊 Densidad", f"{density:.3f}")
    
    with col5:
        avg_clustering = nx.average_clustering(G.to_undirected())
        st.metric("🕸️ Clustering", f"{avg_clustering:.3f}")
    
    st.markdown("---")
    
    # NUEVA SECCIÓN: Tabla de datos del dataset
    st.subheader("📋 Vista de Datos del Dataset")
    st.dataframe(df.head(20), use_container_width=True)
    
    # NUEVA SECCIÓN: Estadísticas detalladas
    st.subheader("📊 Estadísticas Detalladas")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Por Desarrollador:**")
        dev_stats = df.groupby('developer_source').agg({
            'weight': 'sum',
            'developer_target': 'count',
            'repo': 'nunique'
        }).rename(columns={
            'weight': 'Peso Total',
            'developer_target': 'Interacciones',
            'repo': 'Repositorios'
        }).head(10)
        st.dataframe(dev_stats)
    
    with col2:
        st.write("**Por Tipo de Interacción:**")
        interaction_stats = df.groupby('interaction_type').agg({
            'weight': ['count', 'sum', 'mean']
        }).round(2)
        interaction_stats.columns = ['Cantidad', 'Peso Total', 'Peso Promedio']
        st.dataframe(interaction_stats)
    
    # Distribución de datos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribución por Tipo de Interacción")
        interaction_counts = df['interaction_type'].value_counts()
        
        fig = px.pie(
            values=interaction_counts.values,
            names=interaction_counts.index,
            title="Tipos de Interacción en GitHub",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Distribución por Repositorio")
        repo_counts = df['repo'].value_counts().head(10)
        
        fig = px.bar(
            x=repo_counts.values,
            y=repo_counts.index,
            orientation='h',
            title="Top 10 Repositorios por Actividad",
            color=repo_counts.values,
            color_continuous_scale="viridis"
        )
        fig.update_layout(yaxis_title="Repositorio", xaxis_title="Número de Interacciones")
        st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas avanzadas con IA
    st.subheader("🤖 Análisis Inteligente de Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Análisis de diversidad
        unique_pairs = len(df[['developer_source', 'developer_target']].drop_duplicates())
        total_possible = G.number_of_nodes() * (G.number_of_nodes() - 1)
        diversity_score = unique_pairs / max(1, total_possible) * 100
        
        st.metric(
            "🎯 Diversidad de Colaboraciones", 
            f"{diversity_score:.1f}%",
            help="Porcentaje de colaboraciones únicas posibles que existen"
        )
    
    with col2:
        # Intensidad promedio
        avg_intensity = df.groupby(['developer_source', 'developer_target'])['weight'].sum().mean()
        st.metric(
            "⚡ Intensidad Promedio",
            f"{avg_intensity:.1f}",
            help="Peso promedio de colaboraciones entre desarrolladores"
        )
    
    with col3:
        # Factor de reciprocidad
        reciprocity = nx.reciprocity(G)
        st.metric(
            "🔄 Reciprocidad",
            f"{reciprocity:.3f}",
            help="Grado de colaboraciones bidireccionales"
        )
# Mantener las funciones existentes
def show_ai_community_analysis(G, ai_community_detector, df, method):
    """Mostrar análisis de comunidades con IA"""
    st.markdown('<h2 class="section-header">👥 Análisis de Comunidades con IA</h2>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="ai-highlight">🤖 Usando algoritmo: {method.upper()}</div>', unsafe_allow_html=True)
    
    try:
        with st.spinner("🔄 Detectando comunidades con IA..."):
            communities = ai_community_detector.detect_communities(method=method)
            community_stats = ai_community_detector.get_community_stats()
            quality_metrics = ai_community_detector.get_community_quality_metrics()
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏘️ Comunidades", community_stats['num_communities'])
        
        with col2:
            st.metric("📊 Modularidad", f"{quality_metrics['modularity']:.3f}")
        
        with col3:
            st.metric("📈 Cobertura", f"{quality_metrics.get('coverage', 0):.3f}")
        
        with col4:
            st.metric("⚡ Performance", f"{quality_metrics.get('performance', 0):.3f}")
        
        # Visualización de comunidades
        st.subheader("🎨 Visualización de Comunidades IA")
        community_fig = ai_community_detector.visualize_communities()
        st.plotly_chart(community_fig, use_container_width=True)
        
        # NUEVA SECCIÓN: Detalles de cada comunidad
        st.subheader("📋 Detalles por Comunidad")
        
        community_details = {}
        for node, comm_id in communities.items():
            if comm_id not in community_details:
                community_details[comm_id] = []
            community_details[comm_id].append(node)
        
        for comm_id, members in community_details.items():
            with st.expander(f"👥 Comunidad {comm_id} ({len(members)} miembros)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Miembros:**")
                    # Obtener métricas si están disponibles
                    try:
                        from network_analyzer import NetworkAnalyzer
                        analyzer = NetworkAnalyzer(G)
                        metrics = analyzer.calculate_all_metrics()
                        
                        for member in members:
                            pagerank_score = metrics.get('pagerank', {}).get(member, 0)
                            st.write(f"• {member} (PR: {pagerank_score:.3f})")
                    except:
                        for member in members:
                            st.write(f"• {member}")
                
                with col2:
                    # Estadísticas de la comunidad
                    comm_interactions = df[
                        (df['developer_source'].isin(members)) | 
                        (df['developer_target'].isin(members))
                    ]
                    
                    st.metric("🔗 Interacciones Totales", len(comm_interactions))
                    st.metric("📁 Repositorios", comm_interactions['repo'].nunique())
                    
                    if len(comm_interactions) > 0:
                        st.metric("⚖️ Peso Promedio", f"{comm_interactions['weight'].mean():.2f}")
                        
                        # Mostrar repositorios más activos en esta comunidad
                        top_repos = comm_interactions['repo'].value_counts().head(3)
                        st.write("**Top Repositorios:**")
                        for repo, count in top_repos.items():
                            st.write(f"• {repo}: {count} interacciones")
        
        # Optimización de comunidades
        st.subheader("🚀 Optimización de Comunidades")
        
        if st.button("🤖 Ejecutar Optimización Genética"):
            with st.spinner("🧬 Optimizando estructura de comunidades..."):
                optimization_results = ai_community_detector.optimize_community_structure()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "📊 Modularidad Original",
                    f"{optimization_results['original_modularity']:.4f}"
                )
            
            with col2:
                if optimization_results['improvement_achieved']:
                    improvement = optimization_results['modularity_improvement']
                    st.metric(
                        "📈 Mejora Conseguida",
                        f"+{improvement:.4f}",
                        delta=f"{improvement:.4f}"
                    )
                else:
                    st.metric("📈 Mejora Conseguida", "No significativa")
            
            st.success("✅ Optimización completada con algoritmo genético")
    
    except Exception as e:
        st.error(f"❌ Error en análisis de comunidades: {str(e)}")
        st.info("💡 Intenta con un método diferente o verifica las dependencias")
def show_collaboration_patterns(ai_optimizer):
    """Mostrar patrones de colaboración detectados por IA"""
    st.markdown('<h2 class="section-header">📊 Patrones Colaborativos con IA</h2>', unsafe_allow_html=True)
    
    with st.spinner("🔄 Detectando patrones con algoritmos de IA..."):
        patterns = ai_optimizer.detect_collaboration_patterns()
    
    # Patrones temporales
    st.subheader("⏰ Patrones Temporales")
    
    temporal_patterns = patterns['temporal_patterns']
    
    if 'peak_hours' in temporal_patterns and temporal_patterns['peak_hours']:
        col1, col2 = st.columns(2)
        
        with col1:
            # Horas pico
            hours_df = pd.DataFrame(list(temporal_patterns['peak_hours'].items()), 
                                  columns=['Hora', 'Colaboraciones'])
            
            fig = px.bar(hours_df, x='Hora', y='Colaboraciones', 
                        title="Horas Pico de Colaboración")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Días activos
            if 'active_days' in temporal_patterns:
                days_df = pd.DataFrame(list(temporal_patterns['active_days'].items()), 
                                     columns=['Día', 'Colaboraciones'])
                
                fig = px.pie(days_df, values='Colaboraciones', names='Día',
                            title="Distribución por Día de la Semana")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📅 No hay datos temporales disponibles para análisis de patrones")
    
    # Patrones por repositorio
    st.subheader("📁 Patrones por Repositorio")
    
    repo_patterns = patterns['repository_patterns']
    
    if repo_patterns:
        repo_stats_list = []
        for repo, stats in repo_patterns.items():
            repo_stats_list.append({
                'Repositorio': repo,
                'Colaboraciones': stats['total_collaborations'],
                'Desarrolladores': stats['unique_developers'],
                'Peso Promedio': stats['avg_weight'],
                'Densidad': stats['collaboration_density']
            })
        
        repo_stats_df = pd.DataFrame(repo_stats_list)
        
        fig = px.scatter(repo_stats_df, 
                        x='Desarrolladores', 
                        y='Colaboraciones',
                        size='Peso Promedio',
                        color='Densidad',
                        hover_name='Repositorio',
                        title="Análisis de Repositorios - Desarrolladores vs Colaboraciones")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Patrones de influencia
    st.subheader("👑 Patrones de Influencia")
    
    influence_patterns = patterns['influence_patterns']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏆 Top Influenciadores (PageRank):**")
        for dev, score in influence_patterns['top_influencers']:
            st.write(f"• {dev}: {score:.4f}")
    
    with col2:
        st.write("**🌉 Top Constructores de Puentes:**")
        for dev, score in influence_patterns['bridge_builders']:
            st.write(f"• {dev}: {score:.4f}")

def show_ai_recommendations(G, metrics, ai_optimizer, df):
    """Mostrar recomendaciones generadas por IA"""
    st.markdown('<h2 class="section-header">📋 Recomendaciones Inteligentes</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="ai-highlight">🎯 Recomendaciones generadas por algoritmos de Machine Learning para optimizar la colaboración</div>', unsafe_allow_html=True)
    
    # Generar recomendaciones
    with st.spinner("🔄 Generando recomendaciones con IA..."):
        recommendations = ai_optimizer.recommend_collaborations(top_k=10)
        team_optimization = ai_optimizer.optimize_team_formation(team_size=4, n_teams=3)
    
    # Recomendaciones de colaboración
    st.subheader("🤝 Nuevas Colaboraciones Recomendadas")
    
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        display_df = rec_df[['developer_1', 'developer_2', 'composite_score', 'similarity_score', 'reason']]
        display_df.columns = ['Desarrollador 1', 'Desarrollador 2', 'Score IA', 'Similitud', 'Razón']
        
        st.dataframe(
            display_df.style.background_gradient(subset=['Score IA']).format({
                'Score IA': '{:.3f}',
                'Similitud': '{:.3f}'
            }),
            use_container_width=True
        )
    
    # Formación de equipos
    st.subheader("👥 Equipos Recomendados")
    
    for team_name, members in team_optimization['balanced_teams'].items():
        with st.expander(f"👥 {team_name} ({len(members)} miembros)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Miembros:**")
                for member in members:
                    pagerank_score = metrics['pagerank'].get(member, 0)
                    st.write(f"• {member} (PR: {pagerank_score:.3f})")
            
            with col2:
                team_metrics = team_optimization['balanced_metrics']['team_diversity_scores'].get(team_name, {})
                if team_metrics:
                    st.metric("Conexiones Internas", team_metrics.get('internal_connections', 0))
                    st.metric("Fuerza Promedio", f"{team_metrics.get('avg_connection_strength', 0):.2f}")

if __name__ == "__main__":
    main()