import pandas as pd
import random
import datetime
from itertools import combinations

def generate_large_github_dataset(n_developers=25, n_repos=8, n_interactions=400):
    """
    Genera un dataset más grande y realista para ver mejor la optimización
    """
    
    # Desarrolladores más realistas
    developers = [
        'alice_martinez', 'bob_wilson', 'charlie_chen', 'diana_rodriguez', 'eve_johnson',
        'frank_smith', 'grace_kim', 'henry_brown', 'ida_taylor', 'jack_davis',
        'kate_anderson', 'liam_garcia', 'maya_thompson', 'noah_white', 'olivia_lee',
        'peter_jackson', 'quinn_miller', 'rosa_lopez', 'sam_williams', 'tina_moore',
        'alex_clark', 'jordan_lewis', 'casey_hall', 'morgan_young', 'riley_king'
    ][:n_developers]
    
    # Repositorios más diversos
    repos = [
        'frontend-react-app', 'backend-api-service', 'mobile-flutter-app', 
        'data-analytics-pipeline', 'auth-microservice', 'notification-system',
        'ml-recommendation-engine', 'devops-infrastructure'
    ][:n_repos]
    
    # Tipos de interacción
    interaction_types = ['commit_review', 'pull_request', 'issue_comment', 'code_review', 'merge_request']
    
    # Generar colaboraciones de forma más realista
    data = []
    
    # Crear algunos "equipos" para simular colaboraciones naturales
    teams = {
        'frontend': developers[:6],
        'backend': developers[5:12], 
        'mobile': developers[10:16],
        'data': developers[15:20],
        'devops': developers[18:]
    }
    
    repo_teams = {
        'frontend-react-app': 'frontend',
        'backend-api-service': 'backend',
        'mobile-flutter-app': 'mobile',
        'data-analytics-pipeline': 'data',
        'auth-microservice': 'backend',
        'notification-system': 'backend',
        'ml-recommendation-engine': 'data',
        'devops-infrastructure': 'devops'
    }
    
    # Generar interacciones dentro de equipos (80% del tiempo)
    for _ in range(int(n_interactions * 0.8)):
        repo = random.choice(repos)
        team_name = repo_teams.get(repo, 'frontend')
        team_members = teams[team_name]
        
        if len(team_members) >= 2:
            source, target = random.sample(team_members, 2)
            
            data.append({
                'developer_source': source,
                'developer_target': target,
                'interaction_type': random.choice(interaction_types),
                'repo': repo,
                'weight': random.randint(1, 8),
                'timestamp': datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 180))
            })
    
    # Generar algunas interacciones cruzadas entre equipos (20% del tiempo)
    for _ in range(int(n_interactions * 0.2)):
        repo = random.choice(repos)
        source = random.choice(developers)
        target = random.choice([d for d in developers if d != source])
        
        data.append({
            'developer_source': source,
            'developer_target': target,
            'interaction_type': random.choice(interaction_types),
            'repo': repo,
            'weight': random.randint(1, 5),
            'timestamp': datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 180))
        })
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Eliminar duplicados y auto-loops
    df = df[df['developer_source'] != df['developer_target']]
    df = df.drop_duplicates(subset=['developer_source', 'developer_target', 'interaction_type', 'repo'])
    
    # Guardar el archivo
    df.to_csv('github_collaboration_data_large.csv', index=False)
    
    print(f"✅ Dataset generado exitosamente!")
    print(f"📊 {len(df)} interacciones")
    print(f"👥 {len(set(df['developer_source']) | set(df['developer_target']))} desarrolladores únicos")
    print(f"📁 {df['repo'].nunique()} repositorios")
    print(f"🔄 {df['interaction_type'].nunique()} tipos de interacciones")
    
    return df

# Generar el dataset
if __name__ == "__main__":
    large_df = generate_large_github_dataset(n_developers=25, n_repos=8, n_interactions=400)
    print("\n📋 Muestra del nuevo dataset:")
    print(large_df.head(10))
    
    print("\n📊 Estadísticas:")
    print(f"Desarrolladores únicos: {len(set(large_df['developer_source']) | set(large_df['developer_target']))}")
    print(f"Interacciones totales: {len(large_df)}")
    print(f"Peso promedio: {large_df['weight'].mean():.2f}")
    print(f"Repositorios: {large_df['repo'].nunique()}")