import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import streamlit as st

class NetworkAnalyzer:
    """Clase para realizar análisis de métricas de red"""
    
    def __init__(self, graph: nx.DiGraph):
        """
        Inicializa el analizador con un grafo
        
        Args:
            graph: Grafo dirigido de NetworkX
        """
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
    
    @st.cache_data
    def calculate_all_metrics(_self) -> Dict[str, Dict[str, float]]:
        """
        Calcula todas las métricas de centralidad para la red
        
        Returns:
            Diccionario con todas las métricas calculadas
        """
        metrics = {}
        
        # PageRank - Identifica nodos influyentes
        metrics['pagerank'] = nx.pagerank(_self.graph, weight='weight')
        
        # Betweenness Centrality - Identifica nodos puente
        metrics['betweenness'] = nx.betweenness_centrality(_self.graph, weight='weight')
        
        # Closeness Centrality - Mide qué tan cerca está un nodo de todos los demás
        metrics['closeness'] = nx.closeness_centrality(_self.graph, distance='weight')
        
        # Eigenvector Centrality - Mide la influencia basada en conexiones influyentes
        try:
            metrics['eigenvector'] = nx.eigenvector_centrality(_self.graph, weight='weight', max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            # Si no converge, usar la versión sin peso
            metrics['eigenvector'] = nx.eigenvector_centrality(_self.graph, max_iter=1000)
        
        # Degree Centrality (in y out para grafos dirigidos)
        metrics['in_degree'] = dict(_self.graph.in_degree(weight='weight'))
        metrics['out_degree'] = dict(_self.graph.out_degree(weight='weight'))
        
        # Degree Centrality para grafo no dirigido
        metrics['degree'] = dict(_self.undirected_graph.degree(weight='weight'))
        
        return metrics
    
    def get_top_nodes(self, metric_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Obtiene los top k nodos según una métrica específica
        
        Args:
            metric_name: Nombre de la métrica
            top_k: Número de nodos a retornar
            
        Returns:
            Lista de tuplas (nodo, valor_métrica)
        """
        metrics = self.calculate_all_metrics()
        
        if metric_name not in metrics:
            raise ValueError(f"Métrica {metric_name} no disponible")
        
        sorted_nodes = sorted(
            metrics[metric_name].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_nodes[:top_k]
    
    def detect_structural_holes(self, threshold: float = 0.1) -> List[str]:
        """
        Detecta nodos que llenan 'agujeros estructurales' en la red
        (alta intermediación, conexiones a grupos desconectados)
        
        Args:
            threshold: Umbral mínimo de intermediación
            
        Returns:
            Lista de nodos que actúan como puentes estructurales
        """
        metrics = self.calculate_all_metrics()
        betweenness = metrics['betweenness']
        
        structural_holes = [
            node for node, bet in betweenness.items() 
            if bet > threshold
        ]
        
        return structural_holes
    
    def calculate_network_properties(self) -> Dict[str, float]:
        """
        Calcula propiedades globales de la red
        
        Returns:
            Diccionario con propiedades de la red
        """
        properties = {}
        
        # Propiedades básicas
        properties['num_nodes'] = self.graph.number_of_nodes()
        properties['num_edges'] = self.graph.number_of_edges()
        properties['density'] = nx.density(self.graph)
        
        # Clustering
        properties['avg_clustering'] = nx.average_clustering(self.undirected_graph, weight='weight')
        
        # Conectividad
        properties['is_connected'] = nx.is_connected(self.undirected_graph)
        properties['num_components'] = nx.number_connected_components(self.undirected_graph)
        
        if properties['is_connected']:
            properties['diameter'] = nx.diameter(self.undirected_graph)
            properties['avg_path_length'] = nx.average_shortest_path_length(self.undirected_graph)
        else:
            properties['diameter'] = None
            properties['avg_path_length'] = None
        
        # Reciprocidad (para grafos dirigidos)
        properties['reciprocity'] = nx.reciprocity(self.graph)
        
        # Asortatividad
        try:
            properties['degree_assortativity'] = nx.degree_assortativity_coefficient(self.undirected_graph)
        except:
            properties['degree_assortativity'] = None
        
        return properties
    
    def identify_key_players(self, top_k: int = 5) -> Dict[str, List[str]]:
        """
        Identifica jugadores clave según diferentes criterios
        
        Args:
            top_k: Número de jugadores por categoría
            
        Returns:
            Diccionario con diferentes tipos de jugadores clave
        """
        metrics = self.calculate_all_metrics()
        
        key_players = {
            'influencers': [node for node, _ in 
                          sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:top_k]],
            'connectors': [node for node, _ in 
                          sorted(metrics['betweenness'].items(), key=lambda x: x[1], reverse=True)[:top_k]],
            'central_players': [node for node, _ in 
                              sorted(metrics['closeness'].items(), key=lambda x: x[1], reverse=True)[:top_k]],
            'highly_connected': [node for node, _ in 
                               sorted(metrics['degree'].items(), key=lambda x: x[1], reverse=True)[:top_k]]
        }
        
        return key_players
    
    def calculate_node_roles(self) -> Dict[str, str]:
        """
        Clasifica nodos según su rol en la red
        
        Returns:
            Diccionario con el rol de cada nodo
        """
        metrics = self.calculate_all_metrics()
        roles = {}
        
        # Normalizar métricas
        pagerank_vals = list(metrics['pagerank'].values())
        betweenness_vals = list(metrics['betweenness'].values())
        degree_vals = list(metrics['degree'].values())
        
        pagerank_threshold = np.percentile(pagerank_vals, 75)
        betweenness_threshold = np.percentile(betweenness_vals, 75)
        degree_threshold = np.percentile(degree_vals, 75)
        
        for node in self.graph.nodes():
            pr = metrics['pagerank'][node]
            bt = metrics['betweenness'][node]
            dg = metrics['degree'][node]
            
            if pr > pagerank_threshold and bt > betweenness_threshold:
                roles[node] = 'Líder Técnico'
            elif bt > betweenness_threshold:
                roles[node] = 'Conector'
            elif pr > pagerank_threshold:
                roles[node] = 'Influyente'
            elif dg > degree_threshold:
                roles[node] = 'Colaborador Activo'
            else:
                roles[node] = 'Desarrollador Regular'
        
        return roles
    
    def analyze_collaboration_patterns(self, df) -> Dict[str, any]:
        """
        Analiza patrones de colaboración basado en los datos originales
        
        Args:
            df: DataFrame con datos de colaboración
            
        Returns:
            Diccionario con análisis de patrones
        """
        patterns = {}
        
        # Análisis por tipo de interacción
        interaction_analysis = {}
        for interaction_type in df['interaction_type'].unique():
            subset = df[df['interaction_type'] == interaction_type]
            
            # Crear subgrafo para este tipo de interacción
            subgraph = nx.DiGraph()
            for _, row in subset.iterrows():
                if subgraph.has_edge(row['developer_source'], row['developer_target']):
                    subgraph[row['developer_source']][row['developer_target']]['weight'] += row['weight']
                else:
                    subgraph.add_edge(row['developer_source'], row['developer_target'], weight=row['weight'])
            
            if subgraph.number_of_edges() > 0:
                interaction_analysis[interaction_type] = {
                    'nodes': subgraph.number_of_nodes(),
                    'edges': subgraph.number_of_edges(),
                    'density': nx.density(subgraph),
                    'top_contributors': sorted(
                        subgraph.out_degree(weight='weight'), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                }
        
        patterns['by_interaction_type'] = interaction_analysis
        
        # Análisis por repositorio
        repo_analysis = {}
        for repo in df['repo'].unique():
            subset = df[df['repo'] == repo]
            developers = set(subset['developer_source']) | set(subset['developer_target'])
            
            repo_analysis[repo] = {
                'total_interactions': len(subset),
                'unique_developers': len(developers),
                'avg_weight': subset['weight'].mean(),
                'interaction_types': subset['interaction_type'].value_counts().to_dict()
            }
        
        patterns['by_repository'] = repo_analysis
        
        return patterns
    
    def find_potential_bottlenecks(self, threshold_ratio: float = 2.0) -> List[Dict[str, any]]:
        """
        Identifica posibles cuellos de botella en la red
        
        Args:
            threshold_ratio: Ratio mínimo entre intermediación y grado
            
        Returns:
            Lista de posibles cuellos de botella
        """
        metrics = self.calculate_all_metrics()
        bottlenecks = []
        
        for node in self.graph.nodes():
            betweenness = metrics['betweenness'][node]
            degree = metrics['degree'][node]
            
            if degree > 0:
                ratio = betweenness / (degree / len(self.graph.nodes()))
                
                if ratio > threshold_ratio and betweenness > 0.05:
                    bottlenecks.append({
                        'node': node,
                        'betweenness': betweenness,
                        'degree': degree,
                        'ratio': ratio,
                        'risk_level': 'Alto' if ratio > 3.0 else 'Medio'
                    })
        
        return sorted(bottlenecks, key=lambda x: x['ratio'], reverse=True)