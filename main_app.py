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
                "🔍 Análisis de Red", 
                "👥 Comunidades IA", 
                "🏆 Líderes Técnicos",
                "🤖 Optimización IA",
                "📊 Patrones Colaborativos",
                "📋 Recomendaciones IA"
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
    elif section == "🔍 Análisis de Red":
        show_network_analysis(G, metrics, visualizer)
    elif section == "👥 Comunidades IA":
        show_ai_community_analysis(G, ai_community_detector, df, community_method)
    elif section == "🏆 Líderes Técnicos":
        show_technical_leaders(metrics, G, df)
    elif section == "🤖 Optimización IA":
        show_ai_optimization(ai_optimizer, optimization_level)
    elif section == "📊 Patrones Colaborativos":
        show_collaboration_patterns(ai_optimizer)
    elif section == "📋 Recomendaciones IA":
        show_ai_recommendations(G, metrics, ai_optimizer, df)

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
    
    # Tabla de datos mejorada
    st.subheader("📋 Vista de Datos Enriquecida")
    
    # Enriquecer datos con métricas
    enriched_df = df.copy()
    enriched_df['source_pagerank'] = enriched_df['developer_source'].map(metrics['pagerank'])
    enriched_df['target_pagerank'] = enriched_df['developer_target'].map(metrics['pagerank'])
    enriched_df['collaboration_strength'] = enriched_df['source_pagerank'] * enriched_df['target_pagerank'] * enriched_df['weight']
    
    # Mostrar con formato mejorado
    display_df = enriched_df[['developer_source', 'developer_target', 'interaction_type', 'repo', 'weight', 'collaboration_strength']].head(20)
    display_df.columns = ['Desarrollador Origen', 'Desarrollador Destino', 'Tipo Interacción', 'Repositorio', 'Peso', 'Fuerza Colaboración']
    
    st.dataframe(
        display_df.style.format({
            'Peso': '{:.0f}',
            'Fuerza Colaboración': '{:.4f}'
        }).background_gradient(subset=['Fuerza Colaboración']),
        use_container_width=True
    )

def show_network_analysis(G, metrics, visualizer):
    """Mostrar análisis detallado de la red"""
    st.markdown('<h2 class="section-header">🔍 Análisis de Red</h2>', unsafe_allow_html=True)
    
    # Métricas de centralidad mejoradas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Top 10 - PageRank")
        pagerank_df = pd.DataFrame([
            {'Desarrollador': dev, 'PageRank': score, 'Rango': i+1}
            for i, (dev, score) in enumerate(sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:10])
        ])
        
        fig = px.bar(
            pagerank_df,
            x='PageRank',
            y='Desarrollador',
            orientation='h',
            title="Desarrolladores más Influyentes (PageRank)",
            color='PageRank',
            color_continuous_scale='viridis',
            text='Rango'
        )
        fig.update_traces(textposition='inside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌉 Top 10 - Intermediación")
        betweenness_df = pd.DataFrame([
            {'Desarrollador': dev, 'Intermediación': score, 'Rango': i+1}
            for i, (dev, score) in enumerate(sorted(metrics['betweenness'].items(), key=lambda x: x[1], reverse=True)[:10])
        ])
        
        fig = px.bar(
            betweenness_df,
            x='Intermediación',
            y='Desarrollador',
            orientation='h',
            title="Desarrolladores Puente (Betweenness)",
            color='Intermediación',
            color_continuous_scale='plasma',
            text='Rango'
        )
        fig.update_traces(textposition='inside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualización de la red con IA
    st.subheader("🕸️ Visualización Inteligente de la Red")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        metric_option = st.selectbox(
            "Métrica para colorear nodos:",
            ['pagerank', 'betweenness', 'closeness', 'eigenvector']
        )
        
        show_labels = st.checkbox("Mostrar etiquetas", value=True)
        node_size_factor = st.slider("Factor tamaño nodos:", 1, 5, 2)
    
    with col1:
        # Crear visualización de red mejorada
        fig = visualizer.create_network_plot(metrics, metric_option)
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis comparativo de métricas
    st.subheader("📈 Análisis Comparativo de Métricas")
    
    comparison_fig = visualizer.create_centrality_comparison(metrics, top_k=8)
    st.plotly_chart(comparison_fig, use_container_width=True)

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
        
        # Análisis detallado por comunidad
        st.subheader("📋 Análisis Detallado por Comunidad")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tamaños de comunidades
            community_sizes = pd.DataFrame([
                {'Comunidad': f'Comunidad {i}', 'Tamaño': size}
                for i, size in community_stats['community_sizes'].items()
            ])
            
            fig = px.bar(
                community_sizes,
                x='Comunidad',
                y='Tamaño',
                title="Distribución del Tamaño de Comunidades",
                color='Tamaño',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Análisis de conexiones inter-comunitarias
            inter_analysis = ai_community_detector.analyze_inter_community_connections(df)
            
            fig = go.Figure(data=[
                go.Bar(name='Intra-comunidad', x=['Conexiones'], y=[inter_analysis['intra_community_count']]),
                go.Bar(name='Inter-comunidad', x=['Conexiones'], y=[inter_analysis['inter_community_count']])
            ])
            fig.update_layout(
                title='Conexiones Intra vs Inter-Comunitarias',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detalles expandibles de comunidades
        st.subheader("🔍 Explorador de Comunidades")
        
        for comm_id in sorted(set(communities.values())):
            members = [node for node, community in communities.items() if community == comm_id]
            
            with st.expander(f"🏘️ Comunidad {comm_id} ({len(members)} miembros)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Miembros:**")
                    st.write(", ".join(members))
                
                with col2:
                    # Estadísticas de la comunidad
                    subgraph = G.subgraph(members)
                    if subgraph.number_of_edges() > 0:
                        density = nx.density(subgraph)
                        st.write(f"**Densidad interna:** {density:.3f}")
                        
                        # Conexiones más fuertes
                        edges_with_weights = [(u, v, data['weight']) for u, v, data in subgraph.edges(data=True)]
                        if edges_with_weights:
                            edges_with_weights.sort(key=lambda x: x[2], reverse=True)
                            st.write("**Top conexiones:**")
                            for u, v, weight in edges_with_weights[:3]:
                                st.write(f"  • {u} ↔ {v}: {weight}")
    
    except Exception as e:
        st.error(f"❌ Error en análisis de comunidades: {str(e)}")
        st.info("💡 Intenta con un método diferente o verifica las dependencias")

def show_technical_leaders(metrics, G, df):
    """Mostrar análisis de líderes técnicos mejorado"""
    st.markdown('<h2 class="section-header">🏆 Líderes Técnicos</h2>', unsafe_allow_html=True)
    
    # Calcular puntuación combinada mejorada
    weights = {
        'pagerank': 0.35,
        'betweenness': 0.30,
        'eigenvector': 0.20,
        'closeness': 0.15
    }
    
    combined_scores = {}
    
    # Normalizar métricas
    for metric_name, metric_dict in metrics.items():
        if metric_name in weights:
            values = list(metric_dict.values())
            max_val = max(values) if values else 1.0
            
            for node, value in metric_dict.items():
                if node not in combined_scores:
                    combined_scores[node] = 0
                combined_scores[node] += (value / max_val) * weights[metric_name]
    
    # Top líderes
    top_leaders = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    st.subheader("🏆 Top 10 Líderes Técnicos")
    
    leaders_df = pd.DataFrame([
        {
            'Desarrollador': leader,
            'Puntuación IA': f"{score:.3f}",
            'PageRank': f"{metrics['pagerank'][leader]:.3f}",
            'Intermediación': f"{metrics['betweenness'][leader]:.3f}",
            'Eigenvector': f"{metrics['eigenvector'][leader]:.3f}",
            'Cercanía': f"{metrics['closeness'][leader]:.3f}",
            'Colaboraciones': len(df[(df['developer_source'] == leader) | (df['developer_target'] == leader)])
        }
        for leader, score in top_leaders
    ])
    
    st.dataframe(
        leaders_df.style.background_gradient(subset=['Puntuación IA']),
        use_container_width=True
    )
    
    # Gráfico de radar mejorado para top 5 líderes
    st.subheader("📊 Perfil Multidimensional - Top 5 Líderes")
    
    top_5_leaders = [leader for leader, _ in top_leaders[:5]]
    
    categories = ['PageRank', 'Intermediación', 'Eigenvector', 'Cercanía', 'Colaboraciones']
    
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, leader in enumerate(top_5_leaders):
        collab_count = len(df[(df['developer_source'] == leader) | (df['developer_target'] == leader)])
        max_collabs = max([len(df[(df['developer_source'] == l) | (df['developer_target'] == l)]) for l in top_5_leaders])
        
        values = [
            metrics['pagerank'][leader] / max(metrics['pagerank'].values()),
            metrics['betweenness'][leader] / max(metrics['betweenness'].values()),
            metrics['eigenvector'][leader] / max(metrics['eigenvector'].values()),
            metrics['closeness'][leader] / max(metrics['closeness'].values()),
            collab_count / max_collabs
        ]
        values += values[:1]  # Cerrar el polígono
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=leader,
            line_color=colors[i],
            fillcolor=colors[i],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title="Perfil de Liderazgo Técnico - Análisis Multidimensional",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_ai_optimization(ai_optimizer, optimization_level):
    """Mostrar optimización con IA"""
    st.markdown('<h2 class="section-header">🤖 Optimización con Inteligencia Artificial</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="optimization-card">🚀 Sistema de Optimización Avanzado - Utilizando algoritmos de Machine Learning para mejorar la colaboración</div>', unsafe_allow_html=True)
    
    # Formación de equipos optimizada
    st.subheader("👥 Formación Óptima de Equipos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team_size = st.slider("Tamaño objetivo por equipo:", 3, 8, 5)
        n_teams = st.slider("Número de equipos:", 2, 6, 3)
    
    with col2:
        if st.button("🤖 Optimizar Formación de Equipos"):
            with st.spinner("🔄 Analizando perfiles y optimizando equipos..."):
                team_results = ai_optimizer.optimize_team_formation(team_size=team_size, n_teams=n_teams)
            
            st.success(f"✅ Equipos optimizados con score de silhouette: {team_results['silhouette_score']:.3f}")
            
            # Mostrar equipos balanceados
            st.subheader("🎯 Equipos Optimizados")
            
            for team_name, members in team_results['balanced_teams'].items():
                with st.expander(f"👥 {team_name} ({len(members)} miembros)"):
                    st.write("**Miembros:**")
                    st.write(", ".join(members))
                    
                    metrics = team_results['balanced_metrics']['team_diversity_scores'][team_name]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🔗 Conexiones Internas", metrics['internal_connections'])
                    with col2:
                        st.metric("⚖️ Fuerza Total", f"{metrics['internal_strength']:.1f}")
                    with col3:
                        st.metric("📊 Fuerza Promedio", f"{metrics['avg_connection_strength']:.2f}")
    
    # Recomendaciones de colaboración
    st.subheader("🎯 Recomendaciones de Colaboración IA")
    
    if st.button("🤖 Generar Recomendaciones"):
        with st.spinner("🔄 Analizando patrones y generando recomendaciones..."):
            recommendations = ai_optimizer.recommend_collaborations(top_k=10)
        
        st.success("✅ Recomendaciones generadas con algoritmos de similitud")
        
        # Mostrar recomendaciones en tabla
        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df[['developer_1', 'developer_2', 'similarity_score', 'mutual_connections', 'composite_score', 'reason']]
        rec_df.columns = ['Desarrollador 1', 'Desarrollador 2', 'Similitud', 'Conexiones Mutuas', 'Score Total', 'Razón']
        
        st.dataframe(
            rec_df.style.background_gradient(subset=['Score Total']).format({
                'Similitud': '{:.3f}',
                'Score Total': '{:.3f}'
            }),
            use_container_width=True
        )
    
    # Optimización estructural
    st.subheader("🏗️ Optimización Estructural de la Red")
    
    if st.button("🚀 Ejecutar Optimización Completa"):
        with st.spinner("🔄 Ejecutando optimización estructural completa..."):
            optimization_results = ai_optimizer.optimize_network_structure()
        
        st.success("✅ Optimización estructural completada")
        
        # Mostrar métricas actuales
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Métricas Actuales")
            current_metrics = optimization_results['current_metrics']
            
            for metric, value in current_metrics.items():
                if isinstance(value, float) and not np.isinf(value):
                    st.metric(metric.replace('_', ' ').title(), f"{value:.3f}")
                else:
                    st.metric(metric.replace('_', ' ').title(), "N/A")
        
        with col2:
            st.subheader("🎯 Oportunidades de Mejora")
            improvements = optimization_results['improvement_opportunities']
            
            for improvement in improvements:
                st.write(f"• {improvement}")

def show_collaboration_patterns(ai_optimizer):
    """Mostrar patrones de colaboración detectados por IA"""
    st.markdown('<h2 class="section-header">📊 Patrones Colaborativos con IA</h2>', unsafe_allow_html=True)
    
    with st.spinner("🔄 Detectando patrones con algoritmos de IA..."):
        patterns = ai_optimizer.detect_collaboration_patterns()
    
    # Patrones temporales
    st.subheader("⏰ Patrones Temporales")
    
    temporal_patterns = patterns['temporal_patterns']
    
    if 'peak_hours' in temporal_patterns:
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
            days_df = pd.DataFrame(list(temporal_patterns['active_days'].items()), 
                                 columns=['Día', 'Colaboraciones'])
            
            fig = px.pie(days_df, values='Colaboraciones', names='Día',
                        title="Distribución por Día de la Semana")
            st.plotly_chart(fig, use_container_width=True)
        
        # Tendencias
        if 'collaboration_trends' in temporal_patterns:
            trends = temporal_patterns['collaboration_trends']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_color = "green" if trends['growth_rate'] > 0 else "red"
                st.metric("📈 Tasa de Crecimiento", 
                         f"{trends['growth_rate']:.2f}",
                         delta=trends['trend'])
            
            with col2:
                st.metric("📊 Volatilidad", f"{trends['volatility']:.2f}")
            
            with col3:
                st.metric("🎯 Tendencia", trends['trend'].title())
    
    # Patrones por repositorio
    st.subheader("📁 Patrones por Repositorio")
    
    repo_patterns = patterns['repository_patterns']
    
    repo_stats_df = pd.DataFrame.from_dict(repo_patterns, orient='index').reset_index()
    repo_stats_df.columns = ['Repositorio', 'Colaboraciones', 'Desarrolladores', 'Peso Promedio', 'Tipos Interacción', 'Densidad']
    
    # Eliminar columna de tipos de interacción para la visualización
    display_cols = ['Repositorio', 'Colaboraciones', 'Desarrolladores', 'Peso Promedio', 'Densidad']
    
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
    
    # Panel de configuración
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Configuración")
        recommendation_type = st.selectbox(
            "Tipo de Recomendación:",
            ["Colaboraciones Nuevas", "Formación de Equipos", "Mejoras Estructurales"]
        )
        
        confidence_level = st.slider("Nivel de Confianza:", 0.5, 1.0, 0.8, 0.1)
    
    with col1:
        if recommendation_type == "Colaboraciones Nuevas":
            st.subheader("🤝 Nuevas Colaboraciones Recomendadas")
            
            recommendations = ai_optimizer.recommend_collaborations(top_k=15)
            
            # Filtrar por nivel de confianza
            filtered_recs = [r for r in recommendations if r['composite_score'] >= confidence_level]
            
            if filtered_recs:
                # Visualización de red de recomendaciones
                rec_graph = nx.Graph()
                for rec in filtered_recs[:10]:
                    rec_graph.add_edge(rec['developer_1'], rec['developer_2'], 
                                     weight=rec['composite_score'])
                
                # Crear visualización
                pos = nx.spring_layout(rec_graph)
                
                edge_x = []
                edge_y = []
                edge_weights = []
                
                for edge in rec_graph.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(edge[2]['weight'])
                
                node_x = [pos[node][0] for node in rec_graph.nodes()]
                node_y = [pos[node][1] for node in rec_graph.nodes()]
                node_text = list(rec_graph.nodes())
                
                fig = go.Figure()
                
                # Aristas
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='rgba(0, 100, 200, 0.6)'),
                    hoverinfo='none',
                    mode='lines'
                ))
                
                # Nodos
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="middle center",
                    hoverinfo='text',
                    marker=dict(
                        size=20,
                        color='lightblue',
                        line=dict(width=2, color="white")
                    )
                ))
                
                fig.update_layout(
                    title="Red de Colaboraciones Recomendadas",
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de recomendaciones
                st.subheader("📋 Detalles de Recomendaciones")
                
                rec_df = pd.DataFrame(filtered_recs[:10])
                display_df = rec_df[['developer_1', 'developer_2', 'composite_score', 'reason']]
                display_df.columns = ['Desarrollador 1', 'Desarrollador 2', 'Score IA', 'Razón']
                
                st.dataframe(
                    display_df.style.background_gradient(subset=['Score IA']),
                    use_container_width=True
                )
            else:
                st.warning("No se encontraron recomendaciones con el nivel de confianza seleccionado")
        
        elif recommendation_type == "Formación de Equipos":
            st.subheader("👥 Equipos Recomendados")
            
            team_results = ai_optimizer.optimize_team_formation(team_size=4, n_teams=3)
            
            for team_name, members in team_results['balanced_teams'].items():
                with st.expander(f"👥 {team_name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Miembros:**")
                        for member in members:
                            pagerank_score = metrics['pagerank'].get(member, 0)
                            st.write(f"• {member} (PR: {pagerank_score:.3f})")
                    
                    with col2:
                        team_metrics = team_results['balanced_metrics']['team_diversity_scores'].get(team_name, {})
                        st.metric("Conexiones Internas", team_metrics.get('internal_connections', 0))
                        st.metric("Fuerza Promedio", f"{team_metrics.get('avg_connection_strength', 0):.2f}")
        
        else:  # Mejoras Estructurales
            st.subheader("🏗️ Mejoras Estructurales Recomendadas")
            
            # Obtener análisis de optimización
            optimization_results = ai_optimizer.optimize_network_structure()
            
            # Mostrar cuellos de botella
            bottlenecks = optimization_results['bottleneck_analysis']
            
            if bottlenecks:
                st.subheader("⚠️ Cuellos de Botella Identificados")
                
                bottleneck_df = pd.DataFrame(bottlenecks)
                bottleneck_df.columns = ['Desarrollador', 'Intermediación', 'Grado', 'Nivel de Riesgo']
                
                st.dataframe(
                    bottleneck_df.style.applymap(
                        lambda x: 'background-color: red' if x == 'Alto' else 'background-color: orange' if x == 'Medio' else '',
                        subset=['Nivel de Riesgo']
                    ),
                    use_container_width=True
                )
            else:
                st.success("✅ No se detectaron cuellos de botella críticos")
            
            # Oportunidades de mejora
            st.subheader("🎯 Plan de Mejora Recomendado")
            
            improvements = optimization_results['improvement_opportunities']
            
            for i, improvement in enumerate(improvements, 1):
                st.write(f"**{i}.** {improvement}")
            
            # Métricas objetivo
            st.subheader("📊 Métricas Objetivo Sugeridas")
            
            current_metrics = optimization_results['current_metrics']
            
            target_improvements = {
                'density': min(0.3, current_metrics['density'] * 1.2),
                'avg_clustering': min(0.8, current_metrics['avg_clustering'] * 1.15),
                'num_components': max(1, current_metrics['num_components'] - 1)
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🎯 Densidad Objetivo",
                    f"{target_improvements['density']:.3f}",
                    delta=f"+{target_improvements['density'] - current_metrics['density']:.3f}"
                )
            
            with col2:
                st.metric(
                    "🎯 Clustering Objetivo", 
                    f"{target_improvements['avg_clustering']:.3f}",
                    delta=f"+{target_improvements['avg_clustering'] - current_metrics['avg_clustering']:.3f}"
                )
            
            with col3:
                st.metric(
                    "🎯 Componentes Objetivo",
                    target_improvements['num_components'],
                    delta=f"{target_improvements['num_components'] - current_metrics['num_components']}"
                )
    
    # Plan de acción con IA
    st.subheader("📅 Plan de Acción Inteligente")
    
    action_plan = generate_ai_action_plan(recommendation_type, G, df, metrics)
    
    for phase in action_plan:
        with st.expander(f"📋 {phase['title']}"):
            st.write(f"**Duración:** {phase['duration']}")
            st.write(f"**Objetivo:** {phase['objective']}")
            st.write("**Acciones:**")
            for action in phase['actions']:
                st.write(f"• {action}")
            
            if 'metrics' in phase:
                st.write("**Métricas a monitorear:**")
                for metric in phase['metrics']:
                    st.write(f"• {metric}")

def generate_ai_action_plan(recommendation_type, G, df, metrics):
    """Genera plan de acción basado en IA"""
    
    if recommendation_type == "Colaboraciones Nuevas":
        return [
            {
                'title': 'Fase 1: Identificación y Contacto Inicial',
                'duration': '1-2 semanas',
                'objective': 'Establecer contacto entre desarrolladores recomendados',
                'actions': [
                    'Presentar desarrolladores con alta compatibilidad',
                    'Organizar sesiones de coffee chat virtuales',
                    'Facilitar presentaciones en reuniones de equipo'
                ],
                'metrics': ['Número de contactos establecidos', 'Feedback inicial de desarrolladores']
            },
            {
                'title': 'Fase 2: Proyectos Piloto',
                'duration': '2-4 semanas', 
                'objective': 'Implementar colaboraciones en proyectos pequeños',
                'actions': [
                    'Asignar tareas colaborativas menores',
                    'Implementar pair programming sessions',
                    'Crear code review cruzado'
                ],
                'metrics': ['Número de colaboraciones activas', 'Calidad del código conjunto']
            },
            {
                'title': 'Fase 3: Evaluación y Expansión',
                'duration': '1-2 semanas',
                'objective': 'Evaluar éxito y expandir colaboraciones',
                'actions': [
                    'Medir satisfacción de desarrolladores',
                    'Analizar métricas de productividad',
                    'Planificar expansión de colaboraciones exitosas'
                ],
                'metrics': ['Score de satisfacción', 'Incremento en métricas de red']
            }
        ]
    
    elif recommendation_type == "Formación de Equipos":
        return [
            {
                'title': 'Fase 1: Análisis de Perfiles',
                'duration': '1 semana',
                'objective': 'Analizar compatibilidad y habilidades complementarias',
                'actions': [
                    'Evaluar skills técnicos de cada desarrollador',
                    'Analizar estilos de trabajo y comunicación',
                    'Identificar roles óptimos dentro de equipos'
                ],
                'metrics': ['Completitud de perfiles', 'Score de compatibilidad']
            },
            {
                'title': 'Fase 2: Formación Gradual',
                'duration': '2-3 semanas',
                'objective': 'Formar equipos gradualmente con proyectos piloto',
                'actions': [
                    'Asignar proyectos pequeños a equipos nuevos',
                    'Facilitar dinámicas de team building',
                    'Establecer canales de comunicación eficientes'
                ],
                'metrics': ['Velocidad de entrega', 'Comunicación interna']
            },
            {
                'title': 'Fase 3: Optimización Continua',
                'duration': 'Continuo',
                'objective': 'Optimizar dinámicas de equipo basado en datos',
                'actions': [
                    'Monitor continuo de métricas de equipo',
                    'Ajustes basados en feedback y performance',
                    'Rotación estratégica si es necesario'
                ],
                'metrics': ['Productividad del equipo', 'Satisfacción de miembros']
            }
        ]
    
    else:  # Mejoras Estructurales
        return [
            {
                'title': 'Fase 1: Diagnóstico Detallado',
                'duration': '1 semana',
                'objective': 'Identificar puntos débiles estructurales',
                'actions': [
                    'Análisis profundo de cuellos de botella',
                    'Identificación de componentes desconectados',
                    'Mapeo de flujos de información críticos'
                ],
                'metrics': ['Número de bottlenecks', 'Componentes aislados']
            },
            {
                'title': 'Fase 2: Intervenciones Estratégicas',
                'duration': '3-4 semanas',
                'objective': 'Implementar cambios estructurales clave',
                'actions': [
                    'Crear conexiones entre componentes aislados',
                    'Diversificar responsabilidades de nodos críticos',
                    'Establecer canales de comunicación redundantes'
                ],
                'metrics': ['Mejora en densidad', 'Reducción de intermediación crítica']
            },
            {
                'title': 'Fase 3: Monitoreo y Ajuste',
                'duration': 'Continuo',
                'objective': 'Mantener estructura optimizada',
                'actions': [
                    'Monitoreo continuo de métricas de red',
                    'Ajustes proactivos ante cambios',
                    'Prevención de nuevos cuellos de botella'
                ],
                'metrics': ['Estabilidad de métricas', 'Resiliencia de la red']
            }
        ]

if __name__ == "__main__":
    main()