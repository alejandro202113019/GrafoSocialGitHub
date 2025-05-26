import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import streamlit as st

class NetworkVisualizer:
    """Clase para visualizar redes usando Plotly"""
    
    def __init__(self, graph: nx.DiGraph):
        """
        Inicializa el visualizador con un grafo
        
        Args:
            graph: Grafo dirigido de NetworkX
        """
        self.graph = graph
        self.pos = None
        self._calculate_layout()
    
    def _calculate_layout(self):
        """Calcula el layout del grafo para visualización"""
        # Usar spring layout para mejor distribución
        self.pos = nx.spring_layout(self.graph, k=1, iterations=50, seed=42)
    
    def create_network_plot(self, metrics: Dict[str, Dict[str, float]], 
                          color_metric: str = 'pagerank',
                          title: Optional[str] = None) -> go.Figure:
        """
        Crea una visualización interactiva de la red
        
        Args:
            metrics: Diccionario con métricas calculadas
            color_metric: Métrica a usar para colorear nodos
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        if color_metric not in metrics:
            color_metric = 'pagerank'
        
        # Preparar datos de nodos
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        node_info = []
        
        for node in self.graph.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Color basado en la métrica seleccionada
            color_value = metrics[color_metric].get(node, 0)
            node_color.append(color_value)
            
            # Tamaño basado en grado
            degree = self.graph.degree(node, weight='weight')
            node_size.append(max(10, min(50, degree * 2)))
            
            # Texto para mostrar
            node_text.append(node)
            
            # Información detallada para hover
            info = f"<b>{node}</b><br>"
            info += f"PageRank: {metrics['pagerank'].get(node, 0):.3f}<br>"
            info += f"Betweenness: {metrics['betweenness'].get(node, 0):.3f}<br>"
            info += f"Closeness: {metrics['closeness'].get(node, 0):.3f}<br>"
            info += f"Conexiones: {self.graph.degree(node, weight='weight'):.1f}"
            node_info.append(info)
        
        # Preparar datos de aristas
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1))
        
        # Crear trazas
        # Aristas
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125, 125, 125, 0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Nodos
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertext=node_info,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                colorbar=dict(
                    title=f"{color_metric.capitalize()}",
                    titleside="right",
                    tickmode="linear",
                    thickness=15
                ),
                line=dict(width=2, color="white"),
                showscale=True
            )
        )
        
        # Crear figura
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title or f'Red de Colaboración GitHub - {color_metric.capitalize()}',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Haz clic y arrastra para mover la vista. Hover para más información.",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def create_degree_distribution_plot(self, metrics: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Crea un gráfico de distribución de grados
        
        Args:
            metrics: Métricas de la red
            
        Returns:
            Figura de Plotly
        """
        degrees = list(metrics['degree'].values())
        
        fig = go.Figure()
        
        # Histograma
        fig.add_trace(go.Histogram(
            x=degrees,
            nbinsx=20,
            name="Distribución",
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Distribución de Grados en la Red',
            xaxis_title='Grado (Peso de Conexiones)',
            yaxis_title='Frecuencia',
            showlegend=False
        )
        
        return fig
    
    def create_centrality_comparison(self, metrics: Dict[str, Dict[str, float]], 
                                   top_k: int = 10) -> go.Figure:
        """
        Crea un gráfico comparando diferentes métricas de centralidad
        
        Args:
            metrics: Métricas de la red
            top_k: Número de nodos top a mostrar
            
        Returns:
            Figura de Plotly
        """
        # Obtener top k nodos por PageRank
        top_nodes = sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:top_k]
        nodes = [node for node, _ in top_nodes]
        
        # Normalizar métricas para comparación
        pagerank_vals = [metrics['pagerank'][node] for node in nodes]
        betweenness_vals = [metrics['betweenness'][node] for node in nodes]
        closeness_vals = [metrics['closeness'][node] for node in nodes]
        eigenvector_vals = [metrics['eigenvector'][node] for node in nodes]
        
        # Normalizar a 0-1
        def normalize(values):
            max_val = max(values) if max(values) > 0 else 1
            return [v / max_val for v in values]
        
        pagerank_norm = normalize(pagerank_vals)
        betweenness_norm = normalize(betweenness_vals)
        closeness_norm = normalize(closeness_vals)
        eigenvector_norm = normalize(eigenvector_vals)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='PageRank',
            x=nodes,
            y=pagerank_norm,
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            name='Betweenness',
            x=nodes,
            y=betweenness_norm,
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            name='Closeness',
            x=nodes,
            y=closeness_norm,
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            name='Eigenvector',
            x=nodes,
            y=eigenvector_norm,
            opacity=0.8
        ))
        
        fig.update_layout(
            title=f'Comparación de Métricas de Centralidad - Top {top_k}',
            xaxis_title='Desarrolladores',
            yaxis_title='Valor Normalizado',
            barmode='group',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_collaboration_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """
        Crea un heatmap de colaboraciones entre desarrolladores
        
        Args:
            df: DataFrame con datos de colaboración
            
        Returns:
            Figura de Plotly
        """
        # Crear matriz de colaboración
        collaboration_matrix = df.pivot_table(
            index='developer_source',
            columns='developer_target',
            values='weight',
            aggfunc='sum',
            fill_value=0
        )
        
        # Asegurar que la matriz sea cuadrada
        all_devs = sorted(set(df['developer_source'].unique()) | set(df['developer_target'].unique()))
        collaboration_matrix = collaboration_matrix.reindex(index=all_devs, columns=all_devs, fill_value=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=collaboration_matrix.values,
            x=collaboration_matrix.columns,
            y=collaboration_matrix.index,
            colorscale='Blues',
            hoverongaps=False,
            hovertemplate='De: %{y}<br>A: %{x}<br>Peso: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Matriz de Colaboración entre Desarrolladores',
            xaxis_title='Desarrollador Destino',
            yaxis_title='Desarrollador Origen',
            width=800,
            height=800
        )
        
        return fig
    
    def create_interaction_timeline(self, df: pd.DataFrame) -> go.Figure:
        """
        Crea un gráfico de línea temporal de interacciones
        
        Args:
            df: DataFrame con datos de colaboración
            
        Returns:
            Figura de Plotly
        """
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            # Si no hay timestamps, crear datos simulados
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            daily_interactions = np.random.poisson(len(df) / 365, len(dates))
            
            timeline_df = pd.DataFrame({
                'date': dates,
                'interactions': daily_interactions
            })
        else:
            # Usar timestamps reales
            df_time = df.copy()
            df_time['date'] = pd.to_datetime(df_time['timestamp']).dt.date
            timeline_df = df_time.groupby('date').size().reset_index(name='interactions')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timeline_df['date'],
            y=timeline_df['interactions'],
            mode='lines+markers',
            name='Interacciones',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Evolución Temporal de Interacciones',
            xaxis_title='Fecha',
            yaxis_title='Número de Interacciones',
            hovermode='x unified'
        )
        
        return fig
    
    def create_repository_network(self, df: pd.DataFrame) -> go.Figure:
        """
        Crea una red de repositorios basada en desarrolladores compartidos
        
        Args:
            df: DataFrame con datos de colaboración
            
        Returns:
            Figura de Plotly
        """
        # Crear grafo de repositorios
        repo_graph = nx.Graph()
        repos = df['repo'].unique()
        
        # Agregar nodos (repositorios)
        for repo in repos:
            repo_graph.add_node(repo)
        
        # Agregar aristas basadas en desarrolladores compartidos
        for i, repo1 in enumerate(repos):
            for repo2 in repos[i+1:]:
                devs1 = set(df[df['repo'] == repo1]['developer_source']) | set(df[df['repo'] == repo1]['developer_target'])
                devs2 = set(df[df['repo'] == repo2]['developer_source']) | set(df[df['repo'] == repo2]['developer_target'])
                
                shared_devs = len(devs1 & devs2)
                if shared_devs > 0:
                    repo_graph.add_edge(repo1, repo2, weight=shared_devs)
        
        # Layout
        pos = nx.spring_layout(repo_graph, k=2, iterations=50)
        
        # Preparar datos
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for repo in repo_graph.nodes():
            x, y = pos[repo]
            node_x.append(x)
            node_y.append(y)
            node_text.append(repo)
            
            # Tamaño basado en número de interacciones
            interactions = len(df[df['repo'] == repo])
            node_size.append(max(20, min(80, interactions)))
        
        # Aristas
        edge_x = []
        edge_y = []
        
        for edge in repo_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Crear figura
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=[f"<b>{repo}</b><br>Interacciones: {len(df[df['repo'] == repo])}" for repo in repo_graph.nodes()],
            marker=dict(
                size=node_size,
                color='lightcoral',
                line=dict(width=2, color="white")
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Red de Repositorios (Conectados por Desarrolladores Compartidos)',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig