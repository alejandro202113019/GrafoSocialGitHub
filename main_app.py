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
from community_detector import CommunityDetector

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Redes Sociales - GitHub",
    page_icon="🔗",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .stDataFrame {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Título principal
    st.markdown('<h1 class="main-header">🔗 Análisis de Redes Sociales en GitHub</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=100)
        st.title("📊 Panel de Control")
        st.markdown("---")
        
        # Selector de sección
        section = st.selectbox(
            "Selecciona una sección:",
            ["📈 Resumen General", "🔍 Análisis de Red", "👥 Comunidades", "🏆 Líderes Técnicos", "📋 Recomendaciones"]
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ Información")
        st.info("Este dashboard analiza las colaboraciones en GitHub usando métricas de redes sociales.")
    
    # Cargar datos
    try:
        df = load_and_process_data()
        G = create_network_graph(df)
        analyzer = NetworkAnalyzer(G)
        visualizer = NetworkVisualizer(G)
        community_detector = CommunityDetector(G)
        
        # Calcular métricas
        metrics = analyzer.calculate_all_metrics()
        communities = community_detector.detect_communities()
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        st.stop()
    
    # Mostrar sección seleccionada
    if section == "📈 Resumen General":
        show_general_overview(df, G, metrics)
    elif section == "🔍 Análisis de Red":
        show_network_analysis(G, metrics, visualizer)
    elif section == "👥 Comunidades":
        show_community_analysis(G, communities, community_detector, df)
    elif section == "🏆 Líderes Técnicos":
        show_technical_leaders(metrics, G, df)
    elif section == "📋 Recomendaciones":
        show_recommendations(G, metrics, communities, df)

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
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    st.markdown("---")
    
    # Distribución de datos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribución por Tipo de Interacción")
        interaction_counts = df['interaction_type'].value_counts()
        
        fig = px.pie(
            values=interaction_counts.values,
            names=interaction_counts.index,
            title="Tipos de Interacción en GitHub"
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
            title="Top 10 Repositorios por Actividad"
        )
        fig.update_layout(yaxis_title="Repositorio", xaxis_title="Número de Interacciones")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de datos
    st.subheader("📋 Vista de Datos")
    st.dataframe(df.head(20), use_container_width=True)

def show_network_analysis(G, metrics, visualizer):
    """Mostrar análisis detallado de la red"""
    st.markdown('<h2 class="section-header">🔍 Análisis de Red</h2>', unsafe_allow_html=True)
    
    # Métricas de centralidad
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Top 10 - PageRank")
        pagerank_df = pd.DataFrame([
            {'Desarrollador': dev, 'PageRank': score}
            for dev, score in sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:10]
        ])
        
        fig = px.bar(
            pagerank_df,
            x='PageRank',
            y='Desarrollador',
            orientation='h',
            title="Desarrolladores más Influyentes (PageRank)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌉 Top 10 - Intermediación")
        betweenness_df = pd.DataFrame([
            {'Desarrollador': dev, 'Intermediación': score}
            for dev, score in sorted(metrics['betweenness'].items(), key=lambda x: x[1], reverse=True)[:10]
        ])
        
        fig = px.bar(
            betweenness_df,
            x='Intermediación',
            y='Desarrollador',
            orientation='h',
            title="Desarrolladores Puente (Betweenness)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualización de la red
    st.subheader("🕸️ Visualización de la Red")
    metric_option = st.selectbox(
        "Selecciona la métrica para colorear los nodos:",
        ['pagerank', 'betweenness', 'closeness', 'eigenvector']
    )
    
    # Crear visualización de red
    fig = visualizer.create_network_plot(metrics, metric_option)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribución de métricas
    st.subheader("📈 Distribución de Métricas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma PageRank
        fig = px.histogram(
            x=list(metrics['pagerank'].values()),
            nbins=20,
            title="Distribución de PageRank"
        )
        fig.update_layout(xaxis_title="PageRank", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Histograma Betweenness
        fig = px.histogram(
            x=list(metrics['betweenness'].values()),
            nbins=20,
            title="Distribución de Intermediación"
        )
        fig.update_layout(xaxis_title="Betweenness", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)

def show_community_analysis(G, communities, community_detector, df):
    """Mostrar análisis de comunidades"""
    st.markdown('<h2 class="section-header">👥 Análisis de Comunidades</h2>', unsafe_allow_html=True)
    
    num_communities = len(set(communities.values()))
    st.metric("🏘️ Número de Comunidades Detectadas", num_communities)
    
    # Visualización de comunidades
    st.subheader("🎨 Visualización de Comunidades")
    fig = community_detector.visualize_communities()
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis por comunidad
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Tamaño de Comunidades")
        community_sizes = defaultdict(int)
        for node, community in communities.items():
            community_sizes[community] += 1
        
        sizes_df = pd.DataFrame([
            {'Comunidad': f'Comunidad {comm}', 'Tamaño': size}
            for comm, size in community_sizes.items()
        ])
        
        fig = px.bar(
            sizes_df,
            x='Comunidad',
            y='Tamaño',
            title="Distribución del Tamaño de Comunidades"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔗 Interacciones entre Comunidades")
        # Crear matriz de interacciones entre comunidades
        df_comm = df.copy()
        df_comm['source_community'] = df_comm['developer_source'].map(communities)
        df_comm['target_community'] = df_comm['developer_target'].map(communities)
        
        # Contar interacciones entre comunidades
        inter_comm = df_comm.groupby(['source_community', 'target_community']).size().reset_index(name='interactions')
        inter_comm = inter_comm[inter_comm['source_community'] != inter_comm['target_community']]
        
        if not inter_comm.empty:
            fig = px.bar(
                inter_comm.head(10),
                x='interactions',
                y=[f"{row['source_community']}→{row['target_community']}" for _, row in inter_comm.head(10).iterrows()],
                orientation='h',
                title="Top Interacciones Entre Comunidades"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Detalles de comunidades
    st.subheader("📋 Detalles de Comunidades")
    
    for comm_id in sorted(set(communities.values())):
        members = [node for node, community in communities.items() if community == comm_id]
        
        with st.expander(f"🏘️ Comunidad {comm_id} ({len(members)} miembros)"):
            st.write("**Miembros:**")
            st.write(", ".join(members))
            
            # Estadísticas de la comunidad
            subgraph = G.subgraph(members)
            if subgraph.number_of_edges() > 0:
                density = nx.density(subgraph)
                st.write(f"**Densidad interna:** {density:.3f}")
                
                # Conexiones más fuertes dentro de la comunidad
                edges_with_weights = [(u, v, data['weight']) for u, v, data in subgraph.edges(data=True)]
                if edges_with_weights:
                    edges_with_weights.sort(key=lambda x: x[2], reverse=True)
                    st.write("**Conexiones más fuertes:**")
                    for u, v, weight in edges_with_weights[:3]:
                        st.write(f"  • {u} ↔ {v}: {weight}")

def show_technical_leaders(metrics, G, df):
    """Mostrar análisis de líderes técnicos"""
    st.markdown('<h2 class="section-header">🏆 Líderes Técnicos</h2>', unsafe_allow_html=True)
    
    # Calcular puntuación combinada para líderes
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
            'Puntuación Combinada': f"{score:.3f}",
            'PageRank': f"{metrics['pagerank'][leader]:.3f}",
            'Intermediación': f"{metrics['betweenness'][leader]:.3f}",
            'Eigenvector': f"{metrics['eigenvector'][leader]:.3f}",
            'Cercanía': f"{metrics['closeness'][leader]:.3f}"
        }
        for leader, score in top_leaders
    ])
    
    st.dataframe(leaders_df, use_container_width=True)
    
    # Gráfico de radar para top 5 líderes
    st.subheader("📊 Perfil de Top 5 Líderes")
    
    top_5_leaders = [leader for leader, _ in top_leaders[:5]]
    
    # Crear gráfico de radar
    categories = ['PageRank', 'Intermediación', 'Eigenvector', 'Cercanía']
    
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, leader in enumerate(top_5_leaders):
        values = [
            metrics['pagerank'][leader] / max(metrics['pagerank'].values()),
            metrics['betweenness'][leader] / max(metrics['betweenness'].values()),
            metrics['eigenvector'][leader] / max(metrics['eigenvector'].values()),
            metrics['closeness'][leader] / max(metrics['closeness'].values())
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
        title="Perfil de Métricas - Top 5 Líderes Técnicos",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de roles por tipo de interacción
    st.subheader("🎭 Análisis de Roles por Interacción")
    
    role_analysis = {}
    for leader, _ in top_leaders[:5]:
        # Interacciones como fuente
        outgoing = df[df['developer_source'] == leader]
        incoming = df[df['developer_target'] == leader]
        
        role_analysis[leader] = {
            'Commits Revisados': len(outgoing[outgoing['interaction_type'] == 'commit_review']),
            'Pull Requests': len(outgoing[outgoing['interaction_type'] == 'pull_request']),
            'Comentarios Issues': len(outgoing[outgoing['interaction_type'] == 'issue_comment']),
            'Colaboraciones Recibidas': len(incoming)
        }
    
    role_df = pd.DataFrame(role_analysis).T
    role_df.index.name = 'Desarrollador'
    
    st.dataframe(role_df, use_container_width=True)

def show_recommendations(G, metrics, communities, df):
    """Mostrar recomendaciones para optimizar la colaboración"""
    st.markdown('<h2 class="section-header">📋 Recomendaciones para Optimización</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Objetivo
    Optimizar la colaboración y flujo de conocimiento en el equipo de desarrollo mediante análisis de redes sociales.
    """)
    
    # 1. Desarrolladores con pocas conexiones
    st.subheader("🔗 1. Desarrolladores que Necesitan Mayor Integración")
    
    degree_centrality = dict(G.degree())
    low_connected = sorted(degree_centrality.items(), key=lambda x: x[1])[:5]
    
    st.warning("**Desarrolladores con pocas conexiones:**")
    for dev, connections in low_connected:
        st.write(f"• **{dev}**: {connections} conexiones")
    
    st.markdown("""
    **💡 Recomendación:** Fomentar la participación de estos desarrolladores en:
    - Revisiones de código cruzadas
    - Sesiones de pair programming
    - Reuniones de planificación técnica
    """)
    
    # 2. Cuellos de botella
    st.subheader("⚠️ 2. Cuellos de Botella en la Comunicación")
    
    # Identificar nodos con alta intermediación pero bajo grado
    bottlenecks = []
    for node in G.nodes():
        betweenness = metrics['betweenness'][node]
        degree = degree_centrality[node]
        
        if betweenness > 0.1 and degree < 5:  # Umbral ajustable
            bottlenecks.append((node, betweenness, degree))
    
    if bottlenecks:
        st.error("**Posibles cuellos de botella detectados:**")
        for node, bet, deg in sorted(bottlenecks, key=lambda x: x[1], reverse=True):
            st.write(f"• **{node}**: Intermediación={bet:.3f}, Conexiones={deg}")
        
        st.markdown("""
        **💡 Recomendación:** Para estos desarrolladores:
        - Distribuir responsabilidades de revisión
        - Crear documentación de procesos
        - Establecer desarrolladores backup
        """)
    else:
        st.success("✅ No se detectaron cuellos de botella críticos")
    
    # 3. Conexiones recomendadas entre comunidades
    st.subheader("🌉 3. Conexiones Estratégicas Entre Comunidades")
    
    # Encontrar pares de desarrolladores de diferentes comunidades sin conexión
    community_pairs = []
    top_pagerank = sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)
    
    for i, (dev1, pr1) in enumerate(top_pagerank[:10]):
        for dev2, pr2 in top_pagerank[i+1:15]:
            if (communities.get(dev1) != communities.get(dev2) and 
                not G.has_edge(dev1, dev2) and not G.has_edge(dev2, dev1)):
                score = pr1 + pr2
                community_pairs.append((dev1, dev2, score, communities.get(dev1), communities.get(dev2)))
    
    if community_pairs:
        st.info("**Conexiones recomendadas entre comunidades:**")
        for dev1, dev2, score, comm1, comm2 in sorted(community_pairs, key=lambda x: x[2], reverse=True)[:5]:
            st.write(f"• **{dev1}** (Comunidad {comm1}) ↔ **{dev2}** (Comunidad {comm2})")
        
        st.markdown("""
        **💡 Recomendación:** Fomentar colaboración mediante:
        - Proyectos transversales
        - Code reviews cruzados
        - Intercambio de conocimiento técnico
        """)
    
    # 4. Métricas de éxito
    st.subheader("📊 4. Métricas para Medir Mejoras")
    
    current_metrics = {
        "Densidad de Red": f"{nx.density(G):.3f}",
        "Clustering Promedio": f"{nx.average_clustering(G.to_undirected()):.3f}",
        "Diámetro de Red": "N/A" if not nx.is_connected(G.to_undirected()) else str(nx.diameter(G.to_undirected())),
        "Desarrolladores Activos": len(G.nodes()),
        "Interacciones Totales": len(df)
    }
    
    metrics_df = pd.DataFrame([
        {"Métrica": k, "Valor Actual": v}
        for k, v in current_metrics.items()
    ])
    
    st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("""
    **🎯 Objetivos de mejora sugeridos:**
    - Aumentar densidad de red en 10-15%
    - Incrementar clustering promedio
    - Reducir desarrolladores aislados a menos de 5%
    - Incrementar interacciones entre comunidades en 20%
    """)
    
    # 5. Plan de acción
    st.subheader("📅 5. Plan de Acción Recomendado")
    
    action_plan = [
        {
            "Fase": "Fase 1 (Semana 1-2)",
            "Acciones": [
                "Identificar desarrolladores con pocas conexiones",
                "Asignar mentores a desarrolladores aislados",
                "Implementar rotación en revisiones de código"
            ]
        },
        {
            "Fase": "Fase 2 (Semana 3-4)",
            "Acciones": [
                "Establecer proyectos transversales entre comunidades",
                "Crear sesiones regulares de knowledge sharing",
                "Implementar pair programming semanal"
            ]
        },
        {
            "Fase": "Fase 3 (Semana 5-8)",
            "Acciones": [
                "Monitorear métricas de red semanalmente",
                "Ajustar estrategias según resultados",
                "Documentar mejores prácticas identificadas"
            ]
        }
    ]
    
    for phase in action_plan:
        with st.expander(f"📋 {phase['Fase']}"):
            for action in phase['Acciones']:
                st.write(f"• {action}")

if __name__ == "__main__":
    main()