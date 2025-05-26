import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import streamlit as st

# Intentar importar python-louvain, si no está disponible usar métodos alternativos
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

class CommunityDetector:
    """Clase para detectar y analizar comunidades en la red"""
    
    def __init__(self, graph: nx.DiGraph):
        """
        Inicializa el detector con un grafo
        
        Args:
            graph: Grafo dirigido de NetworkX
        """
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
        self.communities = None
    
    @st.cache_data
    def detect_communities(_self, method: str = 'louvain') -> Dict[str, int]:
        """
        Detecta comunidades en el grafo
        
        Args:
            method: Método a usar ('louvain', 'greedy', 'girvan_newman')
            
        Returns:
            Diccionario con asignaciones de comunidades {nodo: comunidad_id}
        """
        if method == 'louvain' and LOUVAIN_AVAILABLE:
            _self.communities = community_louvain.best_partition(_self.undirected_graph, weight='weight')
        elif method == 'greedy' or not LOUVAIN_AVAILABLE:
            # Usar algoritmo greedy de NetworkX
            communities_list = list(nx.algorithms.community.greedy_modularity_communities(_self.undirected_graph, weight='weight'))
            _self.communities = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    _self.communities[node] = i
        elif method == 'girvan_newman':
            # Usar Girvan-Newman (más lento pero efectivo)
            communities_iter = nx.algorithms.community.girvan_newman(_self.undirected_graph)
            communities_list = list(next(communities_iter))
            _self.communities = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    _self.communities[node] = i
        else:
            raise ValueError(f"Método {method} no reconocido o no disponible")
        
        return _self.communities
    
    def get_community_stats(self) -> Dict[str, any]:
        """
        Calcula estadísticas de las comunidades detectadas
        
        Returns:
            Diccionario con estadísticas de comunidades
        """
        if self.communities is None:
            self.detect_communities()
        
        stats = {}
        
        # Número de comunidades
        num_communities = len(set(self.communities.values()))
        stats['num_communities'] = num_communities
        
        # Tamaños de comunidades
        community_sizes = defaultdict(int)
        for node, community in self.communities.items():
            community_sizes[community] += 1
        
        stats['community_sizes'] = dict(community_sizes)
        stats['avg_community_size'] = np.mean(list(community_sizes.values()))
        stats['max_community_size'] = max(community_sizes.values())
        stats['min_community_size'] = min(community_sizes.values())
        
        # Modularidad
        stats['modularity'] = nx.algorithms.community.modularity(
            self.undirected_graph, 
            [set(node for node, comm in self.communities.items() if comm == i) 
             for i in range(num_communities)],
            weight='weight'
        )
        
        return stats
    
    def get_community_leaders(self, metrics: Dict[str, Dict[str, float]], 
                            method: str = 'pagerank') -> Dict[int, Dict[str, str]]:
        """
        Identifica líderes en cada comunidad
        
        Args:
            metrics: Métricas de centralidad
            method: Métrica a usar para identificar líderes
            
        Returns:
            Diccionario con líderes por comunidad
        """
        if self.communities is None:
            self.detect_communities()
        
        leaders = {}
        
        for community_id in set(self.communities.values()):
            community_members = [node for node, comm in self.communities.items() 
                               if comm == community_id]
            
            if not community_members:
                continue
            
            # Líder por diferentes métricas
            pagerank_leader = max(community_members, 
                                key=lambda x: metrics['pagerank'].get(x, 0))
            betweenness_leader = max(community_members, 
                                   key=lambda x: metrics['betweenness'].get(x, 0))
            degree_leader = max(community_members, 
                              key=lambda x: metrics['degree'].get(x, 0))
            
            leaders[community_id] = {
                'pagerank_leader': pagerank_leader,
                'betweenness_leader': betweenness_leader,
                'degree_leader': degree_leader,
                'members': community_members,
                'size': len(community_members)
            }
        
        return leaders
    
    def analyze_inter_community_connections(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analiza conexiones entre comunidades
        
        Args:
            df: DataFrame con datos de colaboración
            
        Returns:
            Análisis de conexiones inter-comunitarias
        """
        if self.communities is None:
            self.detect_communities()
        
        # Agregar información de comunidad al DataFrame
        df_comm = df.copy()
        df_comm['source_community'] = df_comm['developer_source'].map(self.communities)
        df_comm['target_community'] = df_comm['developer_target'].map(self.communities)
        
        # Separar interacciones intra e inter comunitarias
        intra_community = df_comm[df_comm['source_community'] == df_comm['target_community']]
        inter_community = df_comm[df_comm['source_community'] != df_comm['target_community']]
        
        analysis = {
            'total_interactions': len(df_comm),
            'intra_community_count': len(intra_community),
            'inter_community_count': len(inter_community),
            'intra_community_ratio': len(intra_community) / len(df_comm),
            'inter_community_ratio': len(inter_community) / len(df_comm)
        }
        
        # Flujo entre comunidades específicas
        if len(inter_community) > 0:
            community_flow = inter_community.groupby(['source_community', 'target_community']).agg({
                'weight': 'sum',
                'interaction_type': 'count'
            }).reset_index()
            community_flow.columns = ['source_community', 'target_community', 'total_weight', 'interaction_count']
            analysis['community_flow'] = community_flow.to_dict('records')
        else:
            analysis['community_flow'] = []
        
        # Análisis por tipo de interacción
        interaction_analysis = {}
        for interaction_type in df_comm['interaction_type'].unique():
            subset = df_comm[df_comm['interaction_type'] == interaction_type]
            intra = subset[subset['source_community'] == subset['target_community']]
            inter = subset[subset['source_community'] != subset['target_community']]
            
            interaction_analysis[interaction_type] = {
                'total': len(subset),
                'intra_community': len(intra),
                'inter_community': len(inter),
                'inter_ratio': len(inter) / len(subset) if len(subset) > 0 else 0
            }
        
        analysis['by_interaction_type'] = interaction_analysis
        
        return analysis
    
    def visualize_communities(self, title: Optional[str] = None) -> go.Figure:
        """
        Crea visualización de comunidades
        
        Args:
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        if self.communities is None:
            self.detect_communities()
        
        # Calcular layout
        pos = nx.spring_layout(self.undirected_graph, k=1, iterations=50, seed=42)
        
        # Preparar datos por comunidad
        community_colors = px.colors.qualitative.Set3
        
        fig = go.Figure()
        
        # Agregar aristas
        edge_x = []
        edge_y = []
        
        for edge in self.undirected_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125, 125, 125, 0.3)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Agregar nodos por comunidad
        for community_id in set(self.communities.values()):
            community_nodes = [node for node, comm in self.communities.items() 
                             if comm == community_id]
            
            node_x = [pos[node][0] for node in community_nodes]
            node_y = [pos[node][1] for node in community_nodes]
            
            # Tamaño basado en grado
            node_sizes = [max(10, min(30, self.undirected_graph.degree(node, weight='weight')))
                         for node in community_nodes]
            
            # Información para hover
            hover_text = [f"<b>{node}</b><br>Comunidad: {community_id}<br>Grado: {self.undirected_graph.degree(node, weight='weight'):.1f}"
                         for node in community_nodes]
            
            color = community_colors[community_id % len(community_colors)]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=community_nodes,
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                hovertext=hover_text,
                hoverinfo='text',
                marker=dict(
                    size=node_sizes,
                    color=color,
                    line=dict(width=1, color="white")
                ),
                name=f'Comunidad {community_id}',
                showlegend=True
            ))
        
        fig.update_layout(
            title=title or 'Detección de Comunidades en la Red de Colaboración',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_community_flow_diagram(self, df: pd.DataFrame) -> go.Figure:
        """
        Crea diagrama de flujo entre comunidades
        
        Args:
            df: DataFrame con datos de colaboración
            
        Returns:
            Figura de Plotly (Sankey diagram)
        """
        analysis = self.analyze_inter_community_connections(df)
        community_flow = analysis.get('community_flow', [])
        
        if not community_flow:
            # Crear figura vacía si no hay flujo entre comunidades
            fig = go.Figure()
            fig.add_annotation(
                text="No hay conexiones entre comunidades suficientes para mostrar",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Preparar datos para Sankey
        communities = set()
        for flow in community_flow:
            communities.add(f"Comunidad {flow['source_community']}")
            communities.add(f"Comunidad {flow['target_community']}")
        
        communities = sorted(list(communities))
        community_indices = {comm: i for i, comm in enumerate(communities)}
        
        sources = []
        targets = []
        values = []
        
        for flow in community_flow:
            source_label = f"Comunidad {flow['source_community']}"
            target_label = f"Comunidad {flow['target_community']}"
            
            sources.append(community_indices[source_label])
            targets.append(community_indices[target_label])
            values.append(flow['interaction_count'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=communities,
                color="blue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
        fig.update_layout(
            title_text="Flujo de Interacciones entre Comunidades",
            font_size=10
        )
        
        return fig
    
    def get_community_quality_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de calidad de las comunidades
        
        Returns:
            Métricas de calidad
        """
        if self.communities is None:
            self.detect_communities()
        
        metrics = {}
        
        # Modularidad
        community_sets = []
        for i in set(self.communities.values()):
            community_set = set(node for node, comm in self.communities.items() if comm == i)
            community_sets.append(community_set)
        
        metrics['modularity'] = nx.algorithms.community.modularity(
            self.undirected_graph, community_sets, weight='weight'
        )
        
        # Cobertura (fracción de aristas intra-comunitarias)
        metrics['coverage'] = nx.algorithms.community.coverage(
            self.undirected_graph, community_sets
        )
        
        # Performance (combinación de cobertura y modularidad)
        metrics['performance'] = nx.algorithms.community.performance(
            self.undirected_graph, community_sets
        )
        
        return metrics
    
    def find_bridge_nodes(self) -> List[Tuple[str, int]]:
        """
        Encuentra nodos que actúan como puentes entre comunidades
        
        Returns:
            Lista de (nodo, número_de_comunidades_conectadas)
        """
        if self.communities is None:
            self.detect_communities()
        
        bridge_nodes = []
        
        for node in self.graph.nodes():
            # Obtener comunidades de vecinos
            neighbor_communities = set()
            for neighbor in self.graph.neighbors(node):
                neighbor_communities.add(self.communities.get(neighbor))
            
            # Si conecta múltiples comunidades, es un puente
            if len(neighbor_communities) > 1:
                bridge_nodes.append((node, len(neighbor_communities)))
        
        # Ordenar por número de comunidades conectadas
        bridge_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return bridge_nodes