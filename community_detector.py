import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import streamlit as st
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Intentar importar python-louvain, si no está disponible usar métodos alternativos
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("Warning: python-louvain not available, using alternative methods")

class AIOptimizedCommunityDetector:
    """Clase mejorada para detectar y optimizar comunidades usando IA"""
    
    def __init__(self, graph: nx.DiGraph):
        """
        Inicializa el detector con un grafo
        
        Args:
            graph: Grafo dirigido de NetworkX
        """
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
        self.communities = None
        self.embedding_matrix = None
        self.optimal_communities = None
    
    def detect_communities(self, method: str = 'hybrid_ai') -> Dict[str, int]:
        """
        Detecta comunidades usando diferentes métodos incluidos algoritmos de IA
        
        Args:
            method: Método a usar ('louvain', 'greedy', 'spectral_ai', 'kmeans_ai', 'hybrid_ai')
            
        Returns:
            Diccionario con asignaciones de comunidades {nodo: comunidad_id}
        """
        if method == 'louvain' and LOUVAIN_AVAILABLE:
            self.communities = community_louvain.best_partition(self.undirected_graph, weight='weight')
        
        elif method == 'greedy' or (method == 'louvain' and not LOUVAIN_AVAILABLE):
            # Usar algoritmo greedy de NetworkX
            communities_list = list(nx.algorithms.community.greedy_modularity_communities(
                self.undirected_graph, weight='weight'))
            self.communities = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    self.communities[node] = i
        
        elif method == 'spectral_ai':
            self.communities = self._spectral_clustering_ai()
        
        elif method == 'kmeans_ai':
            self.communities = self._kmeans_clustering_ai()
        
        elif method == 'hybrid_ai':
            self.communities = self._hybrid_ai_clustering()
        
        else:
            raise ValueError(f"Método {method} no reconocido")
        
        return self.communities
    
    def _create_node_embeddings(self) -> np.ndarray:
        """
        Crea embeddings de nodos usando características de la red
        
        Returns:
            Matrix de embeddings de nodos
        """
        if self.embedding_matrix is not None:
            return self.embedding_matrix
        
        nodes = list(self.undirected_graph.nodes())
        n_nodes = len(nodes)
        
        # Calcular características para cada nodo
        features = []
        
        # Métricas de centralidad
        pagerank = nx.pagerank(self.undirected_graph, weight='weight')
        betweenness = nx.betweenness_centrality(self.undirected_graph, weight='weight')
        closeness = nx.closeness_centrality(self.undirected_graph, distance='weight')
        degree_centrality = nx.degree_centrality(self.undirected_graph)
        
        try:
            eigenvector = nx.eigenvector_centrality(self.undirected_graph, weight='weight', max_iter=1000)
        except:
            eigenvector = {node: 0.0 for node in nodes}
        
        # Clustering coefficient
        clustering = nx.clustering(self.undirected_graph, weight='weight')
        
        for node in nodes:
            node_features = [
                pagerank.get(node, 0),
                betweenness.get(node, 0),
                closeness.get(node, 0),
                degree_centrality.get(node, 0),
                eigenvector.get(node, 0),
                clustering.get(node, 0),
                self.undirected_graph.degree(node, weight='weight') / n_nodes,  # Normalized degree
            ]
            features.append(node_features)
        
        # Agregar características de vecindario
        for i, node in enumerate(nodes):
            neighbors = list(self.undirected_graph.neighbors(node))
            if neighbors:
                # Promedio de características de vecinos
                neighbor_pagerank = np.mean([pagerank.get(n, 0) for n in neighbors])
                neighbor_degree = np.mean([self.undirected_graph.degree(n, weight='weight') for n in neighbors])
                features[i].extend([neighbor_pagerank, neighbor_degree])
            else:
                features[i].extend([0.0, 0.0])
        
        self.embedding_matrix = np.array(features)
        
        # Normalizar características
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.embedding_matrix = scaler.fit_transform(self.embedding_matrix)
        
        return self.embedding_matrix
    
    def _spectral_clustering_ai(self) -> Dict[str, int]:
        """
        Aplica Spectral Clustering usando IA para optimizar parámetros
        
        Returns:
            Diccionario con asignaciones de comunidades
        """
        embeddings = self._create_node_embeddings()
        nodes = list(self.undirected_graph.nodes())
        
        # Encontrar número óptimo de clusters usando silhouette score
        best_n_clusters = self._optimize_n_clusters(embeddings, method='spectral')
        
        # Aplicar Spectral Clustering
        spectral = SpectralClustering(
            n_clusters=best_n_clusters,
            affinity='nearest_neighbors',
            random_state=42,
            n_neighbors=min(10, len(nodes)-1)
        )
        
        cluster_labels = spectral.fit_predict(embeddings)
        
        return {nodes[i]: int(cluster_labels[i]) for i in range(len(nodes))}
    
    def _kmeans_clustering_ai(self) -> Dict[str, int]:
        """
        Aplica K-Means Clustering optimizado con IA
        
        Returns:
            Diccionario con asignaciones de comunidades
        """
        embeddings = self._create_node_embeddings()
        nodes = list(self.undirected_graph.nodes())
        
        # Encontrar número óptimo de clusters
        best_n_clusters = self._optimize_n_clusters(embeddings, method='kmeans')
        
        # Aplicar K-Means
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        return {nodes[i]: int(cluster_labels[i]) for i in range(len(nodes))}
    
    def _hybrid_ai_clustering(self) -> Dict[str, int]:
        """
        Método híbrido que combina múltiples algoritmos de IA
        
        Returns:
            Diccionario con asignaciones de comunidades optimizadas
        """
        embeddings = self._create_node_embeddings()
        nodes = list(self.undirected_graph.nodes())
        
        # Aplicar múltiples métodos
        methods_results = {}
        
        # Spectral Clustering
        try:
            best_n_spectral = self._optimize_n_clusters(embeddings, method='spectral')
            spectral = SpectralClustering(
                n_clusters=best_n_spectral,
                affinity='nearest_neighbors',
                random_state=42,
                n_neighbors=min(10, len(nodes)-1)
            )
            spectral_labels = spectral.fit_predict(embeddings)
            methods_results['spectral'] = spectral_labels
        except:
            pass
        
        # K-Means
        try:
            best_n_kmeans = self._optimize_n_clusters(embeddings, method='kmeans')
            kmeans = KMeans(n_clusters=best_n_kmeans, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(embeddings)
            methods_results['kmeans'] = kmeans_labels
        except:
            pass
        
        # Greedy modularity (como baseline)
        try:
            communities_list = list(nx.algorithms.community.greedy_modularity_communities(
                self.undirected_graph, weight='weight'))
            greedy_labels = np.zeros(len(nodes))
            for i, community in enumerate(communities_list):
                for node in community:
                    node_idx = nodes.index(node)
                    greedy_labels[node_idx] = i
            methods_results['greedy'] = greedy_labels
        except:
            pass
        
        # Si no hay resultados, usar método simple
        if not methods_results:
            return self._simple_clustering()
        
        # Combinar resultados usando voting
        final_labels = self._ensemble_clustering(methods_results, embeddings)
        
        return {nodes[i]: int(final_labels[i]) for i in range(len(nodes))}
    
    def _optimize_n_clusters(self, embeddings: np.ndarray, method: str = 'kmeans') -> int:
        """
        Optimiza el número de clusters usando silhouette score
        
        Args:
            embeddings: Matrix de embeddings
            method: Método de clustering
            
        Returns:
            Número óptimo de clusters
        """
        n_samples = embeddings.shape[0]
        min_clusters = 2
        max_clusters = min(10, n_samples // 2)
        
        if max_clusters <= min_clusters:
            return min_clusters
        
        best_score = -1
        best_n_clusters = min_clusters
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(embeddings)
                elif method == 'spectral':
                    clusterer = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity='nearest_neighbors',
                        random_state=42,
                        n_neighbors=min(10, n_samples-1)
                    )
                    labels = clusterer.fit_predict(embeddings)
                else:
                    continue
                
                # Verificar que haya más de un cluster
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            except:
                continue
        
        return best_n_clusters
    
    def _ensemble_clustering(self, methods_results: Dict[str, np.ndarray], 
                           embeddings: np.ndarray) -> np.ndarray:
        """
        Combina resultados de múltiples métodos de clustering
        
        Args:
            methods_results: Resultados de diferentes métodos
            embeddings: Embeddings originales
            
        Returns:
            Labels finales combinadas
        """
        if len(methods_results) == 1:
            return list(methods_results.values())[0]
        
        # Evaluar cada método por su silhouette score
        method_scores = {}
        for method, labels in methods_results.items():
            try:
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    method_scores[method] = score
                else:
                    method_scores[method] = -1
            except:
                method_scores[method] = -1
        
        # Usar el método con mejor score
        best_method = max(method_scores.items(), key=lambda x: x[1])[0]
        return methods_results[best_method]
    
    def _simple_clustering(self) -> Dict[str, int]:
        """
        Método de clustering simple como fallback
        
        Returns:
            Diccionario con asignaciones básicas
        """
        nodes = list(self.undirected_graph.nodes())
        
        # Usar grado para clustering simple
        degrees = {node: self.undirected_graph.degree(node, weight='weight') for node in nodes}
        
        # Dividir en 3 grupos por grado
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1])
        n_nodes = len(sorted_nodes)
        
        communities = {}
        for i, (node, _) in enumerate(sorted_nodes):
            if i < n_nodes // 3:
                communities[node] = 0
            elif i < 2 * n_nodes // 3:
                communities[node] = 1
            else:
                communities[node] = 2
        
        return communities
    
    def optimize_community_structure(self) -> Dict[str, any]:
        """
        Optimiza la estructura de comunidades usando algoritmos genéticos
        
        Returns:
            Resultados de la optimización
        """
        if self.communities is None:
            self.detect_communities()
        
        # Implementar optimización genética simple
        best_partition = self._genetic_optimization()
        
        # Calcular métricas de calidad
        quality_metrics = self.get_community_quality_metrics()
        
        optimization_results = {
            'original_modularity': quality_metrics['modularity'],
            'optimized_partition': best_partition,
            'optimization_method': 'Genetic Algorithm',
            'improvement_achieved': True if best_partition else False
        }
        
        if best_partition:
            # Actualizar comunidades con la versión optimizada
            original_communities = self.communities.copy()
            self.communities = best_partition
            optimized_metrics = self.get_community_quality_metrics()
            
            optimization_results['optimized_modularity'] = optimized_metrics['modularity']
            optimization_results['modularity_improvement'] = (
                optimized_metrics['modularity'] - quality_metrics['modularity']
            )
            
            # Restaurar comunidades originales
            self.communities = original_communities
        
        return optimization_results
    
    def _genetic_optimization(self, generations: int = 50, population_size: int = 20) -> Optional[Dict[str, int]]:
        """
        Optimización genética para mejorar la estructura de comunidades
        
        Args:
            generations: Número de generaciones
            population_size: Tamaño de la población
            
        Returns:
            Mejor partición encontrada
        """
        if not self.communities:
            return None
        
        nodes = list(self.communities.keys())
        current_partition = self.communities.copy()
        best_modularity = self._calculate_modularity(current_partition)
        best_partition = current_partition.copy()
        
        # Generar población inicial
        population = [self._mutate_partition(current_partition) for _ in range(population_size)]
        
        for generation in range(generations):
            # Evaluar fitness (modularidad)
            fitness_scores = []
            for partition in population:
                modularity = self._calculate_modularity(partition)
                fitness_scores.append(modularity)
            
            # Seleccionar mejores individuos
            sorted_indices = np.argsort(fitness_scores)[::-1]
            top_performers = [population[i] for i in sorted_indices[:population_size//2]]
            
            # Actualizar mejor solución
            if fitness_scores[sorted_indices[0]] > best_modularity:
                best_modularity = fitness_scores[sorted_indices[0]]
                best_partition = population[sorted_indices[0]].copy()
            
            # Generar nueva población
            new_population = top_performers.copy()
            
            # Crossover y mutación
            while len(new_population) < population_size:
                parent1 = np.random.choice(top_performers)
                parent2 = np.random.choice(top_performers)
                child = self._crossover_partitions(parent1, parent2)
                child = self._mutate_partition(child, mutation_rate=0.1)
                new_population.append(child)
            
            population = new_population
        
        # Retornar mejor partición si hay mejora significativa
        improvement = best_modularity - self._calculate_modularity(current_partition)
        if improvement > 0.01:  # Mejora mínima del 1%
            return best_partition
        
        return None
    
    def _calculate_modularity(self, partition: Dict[str, int]) -> float:
        """
        Calcula la modularidad de una partición
        
        Args:
            partition: Diccionario de asignaciones de comunidades
            
        Returns:
            Valor de modularidad
        """
        try:
            community_sets = []
            for i in set(partition.values()):
                community_set = set(node for node, comm in partition.items() if comm == i)
                if community_set:
                    community_sets.append(community_set)
            
            if not community_sets:
                return 0.0
            
            return nx.algorithms.community.modularity(
                self.undirected_graph, community_sets, weight='weight'
            )
        except:
            return 0.0
    
    def _mutate_partition(self, partition: Dict[str, int], mutation_rate: float = 0.2) -> Dict[str, int]:
        """
        Muta una partición cambiando asignaciones de comunidades
        
        Args:
            partition: Partición original
            mutation_rate: Tasa de mutación
            
        Returns:
            Partición mutada
        """
        mutated = partition.copy()
        nodes = list(partition.keys())
        communities = list(set(partition.values()))
        
        n_mutations = max(1, int(len(nodes) * mutation_rate))
        
        for _ in range(n_mutations):
            node = np.random.choice(nodes)
            new_community = np.random.choice(communities)
            mutated[node] = new_community
        
        return mutated
    
    def _crossover_partitions(self, parent1: Dict[str, int], parent2: Dict[str, int]) -> Dict[str, int]:
        """
        Realiza crossover entre dos particiones
        
        Args:
            parent1: Primera partición padre
            parent2: Segunda partición padre
            
        Returns:
            Partición hijo
        """
        child = {}
        nodes = list(parent1.keys())
        
        # Crossover punto único
        crossover_point = len(nodes) // 2
        
        for i, node in enumerate(nodes):
            if i < crossover_point:
                child[node] = parent1[node]
            else:
                child[node] = parent2[node]
        
        return child
    
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
        stats['modularity'] = self._calculate_modularity(self.communities)
        
        return stats
    
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
        metrics['modularity'] = self._calculate_modularity(self.communities)
        
        # Otras métricas
        try:
            community_sets = []
            for i in set(self.communities.values()):
                community_set = set(node for node, comm in self.communities.items() if comm == i)
                if community_set:
                    community_sets.append(community_set)
            
            if community_sets:
                # Cobertura
                metrics['coverage'] = nx.algorithms.community.coverage(
                    self.undirected_graph, community_sets
                )
                
                # Performance
                metrics['performance'] = nx.algorithms.community.performance(
                    self.undirected_graph, community_sets
                )
            else:
                metrics['coverage'] = 0.0
                metrics['performance'] = 0.0
        except:
            metrics['coverage'] = 0.0
            metrics['performance'] = 0.0
        
        return metrics
    
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
            title=title or 'Detección de Comunidades con IA - Red de Colaboración',
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
            'intra_community_ratio': len(intra_community) / len(df_comm) if len(df_comm) > 0 else 0,
            'inter_community_ratio': len(inter_community) / len(df_comm) if len(df_comm) > 0 else 0
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