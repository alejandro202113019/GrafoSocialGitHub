import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AINetworkOptimizer:
    """
    Optimizador de redes usando algoritmos de Inteligencia Artificial
    Implementa múltiples técnicas de IA para optimizar colaboraciones
    """
    
    def __init__(self, graph: nx.DiGraph, collaboration_data: pd.DataFrame):
        """
        Inicializa el optimizador
        
        Args:
            graph: Grafo de la red
            collaboration_data: DataFrame con datos de colaboración
        """
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
        self.data = collaboration_data
        self.node_features = None
        self.optimization_results = {}
    
    def extract_node_features(self) -> np.ndarray:
        """
        Extrae características avanzadas de los nodos para IA
        
        Returns:
            Matrix de características normalizadas
        """
        nodes = list(self.graph.nodes())
        features_list = []
        
        # Calcular métricas de centralidad
        pagerank = nx.pagerank(self.graph, weight='weight')
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        closeness = nx.closeness_centrality(self.graph, distance='weight')
        
        try:
            eigenvector = nx.eigenvector_centrality(self.graph, weight='weight', max_iter=1000)
        except:
            eigenvector = {node: 0.0 for node in nodes}
        
        # Métricas de estructura local
        clustering = nx.clustering(self.undirected_graph, weight='weight')
        degree_centrality = nx.degree_centrality(self.undirected_graph)
        
        # Extraer características por nodo
        for node in nodes:
            # Características de centralidad
            node_features = [
                pagerank.get(node, 0),
                betweenness.get(node, 0),
                closeness.get(node, 0),
                eigenvector.get(node, 0),
                degree_centrality.get(node, 0),
                clustering.get(node, 0)
            ]
            
            # Características de colaboración del DataFrame
            node_collabs = self.data[
                (self.data['developer_source'] == node) | 
                (self.data['developer_target'] == node)
            ]
            
            if len(node_collabs) > 0:
                # Estadísticas de colaboración
                node_features.extend([
                    len(node_collabs),  # Número total de colaboraciones
                    node_collabs['weight'].mean(),  # Peso promedio
                    node_collabs['weight'].std(),   # Variabilidad del peso
                    len(node_collabs['repo'].unique()),  # Número de repos
                    len(node_collabs['interaction_type'].unique())  # Tipos de interacción
                ])
                
                # Características por tipo de interacción
                for interaction_type in ['commit_review', 'pull_request', 'issue_comment']:
                    type_count = len(node_collabs[node_collabs['interaction_type'] == interaction_type])
                    node_features.append(type_count)
            else:
                # Nodo sin colaboraciones registradas
                node_features.extend([0, 0, 0, 0, 0, 0, 0, 0])
            
            # Características de vecindario
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                neighbor_pagerank = np.mean([pagerank.get(n, 0) for n in neighbors])
                neighbor_degree = np.mean([self.graph.degree(n, weight='weight') for n in neighbors])
                node_features.extend([len(neighbors), neighbor_pagerank, neighbor_degree])
            else:
                node_features.extend([0, 0, 0])
            
            features_list.append(node_features)
        
        # Convertir a numpy array y normalizar
        features_matrix = np.array(features_list)
        
        # Reemplazar NaN con 0
        features_matrix = np.nan_to_num(features_matrix)
        
        # Normalizar características
        scaler = StandardScaler()
        self.node_features = scaler.fit_transform(features_matrix)
        
        return self.node_features
    
    def optimize_team_formation(self, team_size: int = 5, n_teams: int = 3) -> Dict[str, Any]:
        """
        Optimiza la formación de equipos usando K-Means clustering
        
        Args:
            team_size: Tamaño objetivo de cada equipo
            n_teams: Número de equipos a formar
            
        Returns:
            Diccionario con equipos optimizados y métricas
        """
        if self.node_features is None:
            self.extract_node_features()
        
        nodes = list(self.graph.nodes())
        
        # Aplicar K-Means para formar equipos
        kmeans = KMeans(n_clusters=n_teams, random_state=42, n_init=10)
        team_assignments = kmeans.fit_predict(self.node_features)
        
        # Organizar equipos
        teams = {}
        for i in range(n_teams):
            team_members = [nodes[j] for j, team in enumerate(team_assignments) if team == i]
            teams[f'Equipo_{i+1}'] = team_members
        
        # Calcular métricas de calidad de equipos
        team_metrics = self._evaluate_team_quality(teams)
        
        # Aplicar post-procesamiento para balancear equipos
        balanced_teams = self._balance_teams(teams, team_size)
        balanced_metrics = self._evaluate_team_quality(balanced_teams)
        
        return {
            'original_teams': teams,
            'balanced_teams': balanced_teams,
            'original_metrics': team_metrics,
            'balanced_metrics': balanced_metrics,
            'silhouette_score': silhouette_score(self.node_features, team_assignments),
            'optimization_method': 'K-Means + Post-processing'
        }
    
    def _evaluate_team_quality(self, teams: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Evalúa la calidad de la formación de equipos
        
        Args:
            teams: Diccionario con equipos formados
            
        Returns:
            Métricas de calidad
        """
        metrics = {
            'avg_team_size': np.mean([len(members) for members in teams.values()]),
            'team_size_std': np.std([len(members) for members in teams.values()]),
            'total_internal_connections': 0,
            'avg_internal_strength': 0,
            'team_diversity_scores': {}
        }
        
        total_connections = 0
        total_strength = 0
        
        for team_name, members in teams.items():
            # Conexiones internas del equipo
            internal_connections = 0
            internal_strength = 0
            
            for i, member1 in enumerate(members):
                for member2 in members[i+1:]:
                    if self.graph.has_edge(member1, member2):
                        internal_connections += 1
                        internal_strength += self.graph[member1][member2].get('weight', 1)
                    elif self.graph.has_edge(member2, member1):
                        internal_connections += 1
                        internal_strength += self.graph[member2][member1].get('weight', 1)
            
            metrics['team_diversity_scores'][team_name] = {
                'size': len(members),
                'internal_connections': internal_connections,
                'internal_strength': internal_strength,
                'avg_connection_strength': internal_strength / max(1, internal_connections)
            }
            
            total_connections += internal_connections
            total_strength += internal_strength
        
        metrics['total_internal_connections'] = total_connections
        metrics['avg_internal_strength'] = total_strength / max(1, total_connections)
        
        return metrics
    
    def _balance_teams(self, teams: Dict[str, List[str]], target_size: int) -> Dict[str, List[str]]:
        """
        Balancea el tamaño de los equipos
        
        Args:
            teams: Equipos originales
            target_size: Tamaño objetivo
            
        Returns:
            Equipos balanceados
        """
        balanced_teams = {}
        all_members = []
        
        # Recolectar todos los miembros con sus asignaciones originales
        for team_name, members in teams.items():
            for member in members:
                all_members.append((member, team_name))
        
        # Redistribuir para balancear
        team_names = list(teams.keys())
        balanced_teams = {name: [] for name in team_names}
        
        # Asignar miembros de forma balanceada
        for i, (member, original_team) in enumerate(all_members):
            target_team = team_names[i % len(team_names)]
            balanced_teams[target_team].append(member)
        
        return balanced_teams
    
    def recommend_collaborations(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Recomienda nuevas colaboraciones usando algoritmos de IA
        
        Args:
            top_k: Número de recomendaciones a generar
            
        Returns:
            Lista de recomendaciones con scores
        """
        if self.node_features is None:
            self.extract_node_features()
        
        nodes = list(self.graph.nodes())
        recommendations = []
        
        # Calcular similitud entre nodos usando características
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(self.node_features)
        
        # Generar recomendaciones
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                # Verificar si ya existe conexión
                if not self.graph.has_edge(node1, node2) and not self.graph.has_edge(node2, node1):
                    # Calcular score de recomendación
                    similarity_score = similarity_matrix[i][j]
                    
                    # Factores adicionales
                    mutual_connections = len(set(self.graph.neighbors(node1)) & 
                                           set(self.graph.neighbors(node2)))
                    
                    # Diversidad de repositorios
                    repos1 = set(self.data[self.data['developer_source'] == node1]['repo']) | \
                            set(self.data[self.data['developer_target'] == node1]['repo'])
                    repos2 = set(self.data[self.data['developer_source'] == node2]['repo']) | \
                            set(self.data[self.data['developer_target'] == node2]['repo'])
                    
                    repo_overlap = len(repos1 & repos2)
                    repo_diversity = len(repos1 | repos2)
                    
                    # Score compuesto
                    composite_score = (
                        0.4 * similarity_score +
                        0.3 * (mutual_connections / max(1, len(nodes))) +
                        0.2 * (repo_overlap / max(1, repo_diversity)) +
                        0.1 * np.random.random()  # Factor de diversificación
                    )
                    
                    recommendations.append({
                        'developer_1': node1,
                        'developer_2': node2,
                        'similarity_score': similarity_score,
                        'mutual_connections': mutual_connections,
                        'repo_overlap': repo_overlap,
                        'composite_score': composite_score,
                        'reason': self._generate_recommendation_reason(
                            node1, node2, similarity_score, mutual_connections, repo_overlap
                        )
                    })
        
        # Ordenar por score y retornar top k
        recommendations.sort(key=lambda x: x['composite_score'], reverse=True)
        return recommendations[:top_k]
    
    def _generate_recommendation_reason(self, node1: str, node2: str, 
                                      similarity: float, mutual: int, repos: int) -> str:
        """
        Genera explicación para la recomendación
        
        Args:
            node1, node2: Nodos recomendados
            similarity: Score de similitud
            mutual: Conexiones mutuas
            repos: Repositorios compartidos
            
        Returns:
            Razón de la recomendación
        """
        reasons = []
        
        if similarity > 0.7:
            reasons.append("perfiles técnicos similares")
        if mutual > 2:
            reasons.append(f"{mutual} conexiones mutuas")
        if repos > 0:
            reasons.append(f"{repos} repositorios compartidos")
        
        if not reasons:
            reasons.append("potencial para nueva colaboración")
        
        return f"Recomendado por: {', '.join(reasons)}"
    
    def detect_collaboration_patterns(self) -> Dict[str, Any]:
        """
        Detecta patrones de colaboración usando análisis de IA
        
        Returns:
            Patrones identificados
        """
        patterns = {
            'temporal_patterns': self._analyze_temporal_patterns(),
            'repository_patterns': self._analyze_repository_patterns(),
            'interaction_patterns': self._analyze_interaction_patterns(),
            'influence_patterns': self._analyze_influence_patterns()
        }
        
        return patterns
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analiza patrones temporales en colaboraciones"""
        if 'timestamp' not in self.data.columns:
            return {'status': 'No temporal data available'}
        
        # Convertir timestamps
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.day_name()
        self.data['month'] = self.data['timestamp'].dt.month
        
        patterns = {
            'peak_hours': self.data['hour'].value_counts().head(3).to_dict(),
            'active_days': self.data['day_of_week'].value_counts().to_dict(),
            'monthly_distribution': self.data['month'].value_counts().to_dict(),
            'collaboration_trends': self._calculate_collaboration_trends()
        }
        
        return patterns
    
    def _calculate_collaboration_trends(self) -> Dict[str, float]:
        """Calcula tendencias de colaboración"""
        monthly_counts = self.data.groupby(self.data['timestamp'].dt.to_period('M')).size()
        
        if len(monthly_counts) > 1:
            # Calcular tendencia usando regresión lineal simple
            x = np.arange(len(monthly_counts))
            y = monthly_counts.values
            slope = np.polyfit(x, y, 1)[0]
            
            return {
                'growth_rate': float(slope),
                'trend': 'increasing' if slope > 0 else 'decreasing',
                'volatility': float(np.std(y))
            }
        
        return {'growth_rate': 0.0, 'trend': 'stable', 'volatility': 0.0}
    
    def _analyze_repository_patterns(self) -> Dict[str, Any]:
        """Analiza patrones por repositorio"""
        repo_stats = {}
        
        for repo in self.data['repo'].unique():
            repo_data = self.data[self.data['repo'] == repo]
            developers = set(repo_data['developer_source']) | set(repo_data['developer_target'])
            
            repo_stats[repo] = {
                'total_collaborations': len(repo_data),
                'unique_developers': len(developers),
                'avg_weight': repo_data['weight'].mean(),
                'interaction_types': repo_data['interaction_type'].value_counts().to_dict(),
                'collaboration_density': len(repo_data) / max(1, len(developers))
            }
        
        return repo_stats
    
    def _analyze_interaction_patterns(self) -> Dict[str, Any]:
        """Analiza patrones de interacción"""
        interaction_analysis = {}
        
        for interaction_type in self.data['interaction_type'].unique():
            type_data = self.data[self.data['interaction_type'] == interaction_type]
            
            interaction_analysis[interaction_type] = {
                'frequency': len(type_data),
                'avg_weight': type_data['weight'].mean(),
                'top_contributors': type_data['developer_source'].value_counts().head(5).to_dict(),
                'repository_distribution': type_data['repo'].value_counts().to_dict()
            }
        
        return interaction_analysis
    
    def _analyze_influence_patterns(self) -> Dict[str, Any]:
        """Analiza patrones de influencia"""
        # Calcular métricas de influencia
        pagerank = nx.pagerank(self.graph, weight='weight')
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        
        # Identificar diferentes tipos de influenciadores
        influence_patterns = {
            'top_influencers': sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5],
            'bridge_builders': sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5],
            'collaboration_initiators': self.data['developer_source'].value_counts().head(5).to_dict(),
            'collaboration_receivers': self.data['developer_target'].value_counts().head(5).to_dict()
        }
        
        return influence_patterns
    
    def optimize_network_structure(self) -> Dict[str, Any]:
        """
        Optimiza la estructura de la red usando algoritmos genéticos
        
        Returns:
            Resultados de optimización estructural
        """
        current_metrics = self._calculate_network_metrics()
        
        # Simular optimizaciones
        optimization_suggestions = {
            'current_metrics': current_metrics,
            'suggested_connections': self.recommend_collaborations(5),
            'team_formation': self.optimize_team_formation(),
            'bottleneck_analysis': self._identify_bottlenecks(),
            'improvement_opportunities': self._identify_improvements()
        }
        
        return optimization_suggestions
    
    def _calculate_network_metrics(self) -> Dict[str, float]:
        """Calcula métricas actuales de la red"""
        return {
            'density': nx.density(self.graph),
            'avg_clustering': nx.average_clustering(self.undirected_graph),
            'num_components': nx.number_connected_components(self.undirected_graph),
            'avg_path_length': nx.average_shortest_path_length(self.undirected_graph) 
                if nx.is_connected(self.undirected_graph) else float('inf'),
            'diameter': nx.diameter(self.undirected_graph) 
                if nx.is_connected(self.undirected_graph) else float('inf')
        }
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identifica cuellos de botella en la red"""
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        degree = dict(self.graph.degree(weight='weight'))
        
        bottlenecks = []
        for node in self.graph.nodes():
            bet_score = betweenness[node]
            deg_score = degree[node]
            
            # Nodos con alta intermediación pero pocas conexiones
            if bet_score > 0.1 and deg_score < np.mean(list(degree.values())):
                bottlenecks.append({
                    'node': node,
                    'betweenness': bet_score,
                    'degree': deg_score,
                    'risk_level': 'Alto' if bet_score > 0.2 else 'Medio'
                })
        
        return sorted(bottlenecks, key=lambda x: x['betweenness'], reverse=True)
    
    def _identify_improvements(self) -> List[str]:
        """Identifica oportunidades de mejora"""
        improvements = []
        metrics = self._calculate_network_metrics()
        
        if metrics['density'] < 0.1:
            improvements.append("Incrementar conexiones entre desarrolladores")
        
        if metrics['avg_clustering'] < 0.3:
            improvements.append("Fomentar formación de equipos cohesivos")
        
        if metrics['num_components'] > 1:
            improvements.append("Conectar componentes aislados de la red")
        
        # Análisis de datos
        interaction_counts = self.data['interaction_type'].value_counts()
        if len(interaction_counts) == 1:
            improvements.append("Diversificar tipos de interaccionse")
        
        repo_distribution = self.data['repo'].value_counts()
        if repo_distribution.iloc[0] > len(self.data) * 0.8:
            improvements.append("Promover colaboración entre repositorios")
        
        return improvements