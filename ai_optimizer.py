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
        self.original_metrics = None
        self.optimized_graph = None
        self.optimized_metrics = None
    
    def calculate_comprehensive_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calcula métricas comprehensivas para un grafo
        
        Args:
            graph: Grafo a analizar
            
        Returns:
            Diccionario completo de métricas
        """
        metrics = {}
        undirected_g = graph.to_undirected()
        
        # Métricas básicas
        metrics['num_nodes'] = graph.number_of_nodes()
        metrics['num_edges'] = graph.number_of_edges()
        metrics['density'] = nx.density(graph)
        
        # Métricas de centralidad
        metrics['pagerank'] = nx.pagerank(graph, weight='weight')
        metrics['betweenness_centrality'] = nx.betweenness_centrality(graph, weight='weight')
        metrics['degree_centrality'] = nx.degree_centrality(graph)
        metrics['closeness_centrality'] = nx.closeness_centrality(graph, distance='weight')
        
        try:
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(graph, weight='weight', max_iter=1000)
        except:
            metrics['eigenvector_centrality'] = {node: 0.0 for node in graph.nodes()}
        
        # Métricas globales
        metrics['avg_clustering'] = nx.average_clustering(undirected_g, weight='weight')
        metrics['num_components'] = nx.number_connected_components(undirected_g)
        metrics['reciprocity'] = nx.reciprocity(graph)
        
        if nx.is_connected(undirected_g):
            metrics['diameter'] = nx.diameter(undirected_g)
            metrics['avg_path_length'] = nx.average_shortest_path_length(undirected_g)
        else:
            metrics['diameter'] = float('inf')
            metrics['avg_path_length'] = float('inf')
        
        # Métricas avanzadas
        metrics['transitivity'] = nx.transitivity(undirected_g)
        metrics['global_efficiency'] = nx.global_efficiency(undirected_g)
        
        # Rankings de centralidad
        metrics['betweenness_ranking'] = sorted(
            metrics['betweenness_centrality'].items(), 
            key=lambda x: x[1], reverse=True
        )
        
        metrics['degree_ranking'] = sorted(
            metrics['degree_centrality'].items(), 
            key=lambda x: x[1], reverse=True
        )
        
        metrics['pagerank_ranking'] = sorted(
            metrics['pagerank'].items(), 
            key=lambda x: x[1], reverse=True
        )
        
        return metrics
    
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
    
    def apply_optimization_recommendations(self, top_recommendations: int = 5) -> nx.DiGraph:
        """
        Aplica las mejores recomendaciones de optimización al grafo
        
        Args:
            top_recommendations: Número de recomendaciones a aplicar
            
        Returns:
            Grafo optimizado
        """
        # Guardar métricas originales si no se han guardado
        if self.original_metrics is None:
            self.original_metrics = self.calculate_comprehensive_metrics(self.graph)
        
        # Crear copia del grafo para optimización
        optimized_graph = self.graph.copy()
        
        # Obtener recomendaciones de colaboración
        recommendations = self.recommend_collaborations(top_k=top_recommendations * 2)
        
        # Aplicar las mejores recomendaciones
        applied_recommendations = []
        for i, rec in enumerate(recommendations[:top_recommendations]):
            if rec['composite_score'] > 0.5:  # Solo aplicar recomendaciones de alta calidad
                dev1, dev2 = rec['developer_1'], rec['developer_2']
                
                # Calcular peso de la nueva conexión basado en el score
                new_weight = rec['composite_score'] * 3  # Escalar score a peso
                
                if not optimized_graph.has_edge(dev1, dev2):
                    optimized_graph.add_edge(dev1, dev2, weight=new_weight)
                    applied_recommendations.append({
                        'from': dev1,
                        'to': dev2,
                        'weight': new_weight,
                        'reason': rec['reason']
                    })
                else:
                    # Reforzar conexión existente
                    optimized_graph[dev1][dev2]['weight'] += new_weight * 0.5
                    applied_recommendations.append({
                        'from': dev1,
                        'to': dev2,
                        'weight_increase': new_weight * 0.5,
                        'reason': f"Refuerzo: {rec['reason']}"
                    })
        
        # Aplicar optimizaciones estructurales adicionales
        bottlenecks = self._identify_bottlenecks_advanced(optimized_graph)
        
        # Mitigar cuellos de botella críticos
        for bottleneck in bottlenecks[:2]:  # Solo los 2 más críticos
            critical_node = bottleneck['node']
            
            # Encontrar nodos para crear conexiones alternativas
            neighbors = list(optimized_graph.neighbors(critical_node))
            non_neighbors = [n for n in optimized_graph.nodes() if n != critical_node and n not in neighbors]
            
            if len(non_neighbors) >= 2:
                # Crear conexiones entre no-vecinos para reducir dependencia
                selected_nodes = np.random.choice(non_neighbors, min(2, len(non_neighbors)), replace=False)
                for i in range(len(selected_nodes) - 1):
                    node1, node2 = selected_nodes[i], selected_nodes[i + 1]
                    if not optimized_graph.has_edge(node1, node2):
                        optimized_graph.add_edge(node1, node2, weight=2.0)
                        applied_recommendations.append({
                            'from': node1,
                            'to': node2,
                            'weight': 2.0,
                            'reason': f'Mitigación cuello de botella: {critical_node}'
                        })
        
        # Guardar grafo optimizado y calcular métricas
        self.optimized_graph = optimized_graph
        self.optimized_metrics = self.calculate_comprehensive_metrics(optimized_graph)
        
        # Guardar detalles de optimización
        self.optimization_results = {
            'applied_recommendations': applied_recommendations,
            'bottlenecks_mitigated': len(bottlenecks[:2]),
            'new_connections': len([r for r in applied_recommendations if 'weight_increase' not in r]),
            'reinforced_connections': len([r for r in applied_recommendations if 'weight_increase' in r])
        }
        
        return optimized_graph
    
    def _identify_bottlenecks_advanced(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Identifica cuellos de botella avanzados en la red
        
        Args:
            graph: Grafo a analizar
            
        Returns:
            Lista de cuellos de botella ordenados por criticidad
        """
        betweenness = nx.betweenness_centrality(graph, weight='weight')
        degree = dict(graph.degree(weight='weight'))
        
        bottlenecks = []
        for node in graph.nodes():
            bet_score = betweenness[node]
            deg_score = degree[node]
            
            # Calcular criticidad combinada
            if deg_score > 0:
                criticality = bet_score * (1 + 1/deg_score)  # Alta intermediación + pocas conexiones = crítico
                
                if bet_score > 0.05:  # Umbral mínimo de intermediación
                    bottlenecks.append({
                        'node': node,
                        'betweenness': bet_score,
                        'degree': deg_score,
                        'criticality': criticality,
                        'risk_level': 'Alto' if criticality > 0.1 else 'Medio'
                    })
        
        return sorted(bottlenecks, key=lambda x: x['criticality'], reverse=True)
    
    def get_optimization_comparison(self) -> Dict[str, Any]:
        """
        Genera comparación detallada antes/después de optimización
        
        Returns:
            Diccionario con comparación completa
        """
        if self.original_metrics is None or self.optimized_metrics is None:
            return {'error': 'Optimización no ejecutada. Ejecute apply_optimization_recommendations primero.'}
        
        comparison = {
            'metrics_comparison': {},
            'ranking_changes': {},
            'improvement_summary': {},
            'detailed_analysis': {}
        }
        
        # Comparación de métricas globales
        global_metrics = ['density', 'avg_clustering', 'reciprocity', 'num_edges', 'num_components']
        
        for metric in global_metrics:
            before = self.original_metrics.get(metric, 0)
            after = self.optimized_metrics.get(metric, 0)
            
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                change = after - before
                change_pct = (change / before * 100) if before != 0 else 0
                
                comparison['metrics_comparison'][metric] = {
                    'before': before,
                    'after': after,
                    'change': change,
                    'change_percentage': change_pct,
                    'improved': change > 0
                }
        
        # Cambios en rankings de centralidad
        comparison['ranking_changes'] = self._analyze_ranking_changes()
        
        # Resumen de mejoras
        positive_changes = sum(1 for m in comparison['metrics_comparison'].values() if m.get('improved', False))
        total_metrics = len(comparison['metrics_comparison'])
        
        comparison['improvement_summary'] = {
            'metrics_improved': positive_changes,
            'total_metrics': total_metrics,
            'improvement_rate': positive_changes / total_metrics if total_metrics > 0 else 0,
            'new_connections': self.optimization_results.get('new_connections', 0),
            'reinforced_connections': self.optimization_results.get('reinforced_connections', 0),
            'bottlenecks_mitigated': self.optimization_results.get('bottlenecks_mitigated', 0)
        }
        
        # Análisis detallado por desarrollador
        comparison['detailed_analysis'] = self._analyze_developer_impact()
        
        return comparison
    
    def _analyze_ranking_changes(self) -> Dict[str, List[Dict]]:
        """
        Analiza cambios en rankings de centralidad
        
        Returns:
            Diccionario con cambios en rankings
        """
        changes = {}
        
        # Analizar cambios en betweenness centrality
        before_bet_rank = {dev: rank for rank, (dev, _) in enumerate(self.original_metrics['betweenness_ranking'], 1)}
        after_bet_rank = {dev: rank for rank, (dev, _) in enumerate(self.optimized_metrics['betweenness_ranking'], 1)}
        
        betweenness_changes = []
        for dev in self.graph.nodes():
            before_rank = before_bet_rank.get(dev, len(self.graph.nodes()))
            after_rank = after_bet_rank.get(dev, len(self.graph.nodes()))
            rank_change = before_rank - after_rank  # Positivo = mejora
            
            betweenness_changes.append({
                'developer': dev,
                'before_rank': before_rank,
                'after_rank': after_rank,
                'rank_change': rank_change,
                'before_value': self.original_metrics['betweenness_centrality'][dev],
                'after_value': self.optimized_metrics['betweenness_centrality'][dev]
            })
        
        changes['betweenness_centrality'] = sorted(betweenness_changes, key=lambda x: x['rank_change'], reverse=True)
        
        # Analizar cambios en degree centrality
        before_deg_rank = {dev: rank for rank, (dev, _) in enumerate(self.original_metrics['degree_ranking'], 1)}
        after_deg_rank = {dev: rank for rank, (dev, _) in enumerate(self.optimized_metrics['degree_ranking'], 1)}
        
        degree_changes = []
        for dev in self.graph.nodes():
            before_rank = before_deg_rank.get(dev, len(self.graph.nodes()))
            after_rank = after_deg_rank.get(dev, len(self.graph.nodes()))
            rank_change = before_rank - after_rank
            
            degree_changes.append({
                'developer': dev,
                'before_rank': before_rank,
                'after_rank': after_rank,
                'rank_change': rank_change,
                'before_value': self.original_metrics['degree_centrality'][dev],
                'after_value': self.optimized_metrics['degree_centrality'][dev]
            })
        
        changes['degree_centrality'] = sorted(degree_changes, key=lambda x: x['rank_change'], reverse=True)
        
        return changes
    
    def _analyze_developer_impact(self) -> Dict[str, Any]:
        """
        Analiza el impacto de la optimización por desarrollador
        
        Returns:
            Análisis de impacto por desarrollador
        """
        impact_analysis = {
            'most_benefited': [],
            'most_affected': [],
            'new_connections_by_developer': {},
            'centrality_improvements': {}
        }
        
        # Calcular impacto por desarrollador
        developer_impacts = []
        
        for dev in self.graph.nodes():
            # Cambios en centralidad
            bet_before = self.original_metrics['betweenness_centrality'][dev]
            bet_after = self.optimized_metrics['betweenness_centrality'][dev]
            bet_change = bet_after - bet_before
            
            deg_before = self.original_metrics['degree_centrality'][dev]
            deg_after = self.optimized_metrics['degree_centrality'][dev]
            deg_change = deg_after - deg_before
            
            pagerank_before = self.original_metrics['pagerank'][dev]
            pagerank_after = self.optimized_metrics['pagerank'][dev]
            pagerank_change = pagerank_after - pagerank_before
            
            # Score de impacto total
            total_impact = abs(bet_change) + abs(deg_change) + abs(pagerank_change)
            
            developer_impacts.append({
                'developer': dev,
                'betweenness_change': bet_change,
                'degree_change': deg_change,
                'pagerank_change': pagerank_change,
                'total_impact': total_impact,
                'positive_impact': bet_change > 0 or deg_change > 0 or pagerank_change > 0
            })
        
        # Ordenar por impacto
        developer_impacts.sort(key=lambda x: x['total_impact'], reverse=True)
        
        # Identificar más beneficiados y afectados
        impact_analysis['most_benefited'] = [d for d in developer_impacts if d['positive_impact']][:5]
        impact_analysis['most_affected'] = developer_impacts[:10]
        
        # Contar nuevas conexiones por desarrollador
        for rec in self.optimization_results.get('applied_recommendations', []):
            dev1 = rec.get('from')
            dev2 = rec.get('to')
            
            if dev1 and dev2:
                if dev1 not in impact_analysis['new_connections_by_developer']:
                    impact_analysis['new_connections_by_developer'][dev1] = 0
                if dev2 not in impact_analysis['new_connections_by_developer']:
                    impact_analysis['new_connections_by_developer'][dev2] = 0
                
                impact_analysis['new_connections_by_developer'][dev1] += 1
                impact_analysis['new_connections_by_developer'][dev2] += 1
        
        return impact_analysis
    
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
            improvements.append("Diversificar tipos de interacciones")
        
        repo_distribution = self.data['repo'].value_counts()
        if repo_distribution.iloc[0] > len(self.data) * 0.8:
            improvements.append("Promover colaboración entre repositorios")
        
        return improvements
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Genera reporte comprehensivo de análisis y optimización
        
        Returns:
            Reporte completo del análisis
        """
        if self.original_metrics is None:
            self.original_metrics = self.calculate_comprehensive_metrics(self.graph)
        
        report = {
            'executive_summary': self._generate_executive_summary(),
            'original_network_analysis': self._format_network_analysis(self.original_metrics),
            'optimization_recommendations': self.recommend_collaborations(10),
            'team_formation_analysis': self.optimize_team_formation(),
            'collaboration_patterns': self.detect_collaboration_patterns(),
            'bottleneck_analysis': self._identify_bottlenecks_advanced(self.graph),
            'improvement_opportunities': self._identify_improvements()
        }
        
        # Si hay optimización aplicada, incluir comparación
        if self.optimized_metrics is not None:
            report['optimization_results'] = self.get_optimization_comparison()
            report['optimized_network_analysis'] = self._format_network_analysis(self.optimized_metrics)
        
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Genera resumen ejecutivo del análisis"""
        metrics = self.original_metrics or self.calculate_comprehensive_metrics(self.graph)
        
        # Calcular scores de salud de la red
        density_score = min(1.0, metrics['density'] * 5)  # Normalizar densidad
        clustering_score = metrics['avg_clustering']
        reciprocity_score = metrics['reciprocity']
        
        overall_health = (density_score + clustering_score + reciprocity_score) / 3
        
        summary = {
            'network_health_score': overall_health,
            'total_developers': metrics['num_nodes'],
            'total_interactions': metrics['num_edges'],
            'collaboration_density': metrics['density'],
            'key_insights': [],
            'priority_recommendations': []
        }
        
        # Generar insights automáticos
        if metrics['density'] < 0.1:
            summary['key_insights'].append("Red poco densa - oportunidad para más colaboraciones")
            summary['priority_recommendations'].append("Implementar programa de mentoring cruzado")
        
        if metrics['reciprocity'] > 0.7:
            summary['key_insights'].append("Alta reciprocidad - colaboraciones bidireccionales saludables")
        
        if metrics['num_components'] > 1:
            summary['key_insights'].append("Red fragmentada - equipos aislados identificados")
            summary['priority_recommendations'].append("Crear puentes entre equipos aislados")
        
        # Identificar líderes clave
        top_pagerank = max(metrics['pagerank'].items(), key=lambda x: x[1])
        top_betweenness = max(metrics['betweenness_centrality'].items(), key=lambda x: x[1])
        
        summary['key_leaders'] = {
            'most_influential': top_pagerank[0],
            'key_connector': top_betweenness[0]
        }
        
        return summary
    
    def _format_network_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Formatea análisis de red para reporte"""
        return {
            'basic_metrics': {
                'nodes': metrics['num_nodes'],
                'edges': metrics['num_edges'],
                'density': round(metrics['density'], 4),
                'avg_clustering': round(metrics['avg_clustering'], 4),
                'reciprocity': round(metrics['reciprocity'], 4)
            },
            'centrality_leaders': {
                'pagerank_top5': [(dev, round(score, 4)) for dev, score in metrics['pagerank_ranking'][:5]],
                'betweenness_top5': [(dev, round(score, 4)) for dev, score in metrics['betweenness_ranking'][:5]],
                'degree_top5': [(dev, round(score, 4)) for dev, score in metrics['degree_ranking'][:5]]
            },
            'structural_properties': {
                'components': metrics['num_components'],
                'diameter': metrics['diameter'] if metrics['diameter'] != float('inf') else 'N/A',
                'avg_path_length': round(metrics['avg_path_length'], 4) if metrics['avg_path_length'] != float('inf') else 'N/A',
                'transitivity': round(metrics.get('transitivity', 0), 4),
                'global_efficiency': round(metrics.get('global_efficiency', 0), 4)
            }
        }