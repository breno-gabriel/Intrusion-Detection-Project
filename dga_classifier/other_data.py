import pandas as pd
import random

def generate_database(): 
    # Lendo o arquivo CSV com dados benignos
    benign_data = pd.read_csv('majestic_million.csv')

    # Especificando a coluna correta para os nomes de domínio nos dados benignos
    benign_domains = benign_data.iloc[:, 2].tolist()

    # Lendo o arquivo de texto com dados malignos
    with open('domínios_malignos.txt', 'r') as file:
        malicious_data = file.read().splitlines()

    # Especificando a coluna correta para os nomes de domínio nos dados malignos
    malicious_domains = [line.split(',')[0] for line in malicious_data]

    print(malicious_domains[0])

    # Definindo o tamanho desejado para a base de dados (metade benigno, metade maligno)
    total_samples = 337500
    half_samples = total_samples // 2

    # Amostrando aleatoriamente os dados benignos
    sampled_benign_data = random.sample(benign_domains, half_samples)

    # Amostrando aleatoriamente os dados malignos
    sampled_malicious_data = random.sample(malicious_domains, half_samples)

    # Criando a base de dados final
    domains = sampled_benign_data + sampled_malicious_data
    labels = ['benigno'] * half_samples + ['maligno'] * half_samples

    # Misturando aleatoriamente os dados
    combined_data = list(zip(domains, labels))
    random.shuffle(combined_data)
    domains, labels = zip(*combined_data)

    return domains, labels


