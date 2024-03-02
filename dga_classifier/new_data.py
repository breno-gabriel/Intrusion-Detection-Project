# import csv
# import pickle
# import os

# def get_data(num_per_dga=10000, force=False):
#     data_file = 'mydata.pkl'

#     # if force or (not os.path.isfile(data_file)):
#     domains = []
#     labels = []

#     dga_families = ['necurs', 'rovnix', 'fobber', 'ranbyus', 'cryptolocker', 'pykspa', 'corebot', 'kraken',
#                         'nymaim', 'dircrypt', 'symmi', 'pushdo', 'simda', 'qadars', 'ramdo', 'suppobox', 'conficker',
#                         'matsnu', 'murofet', 'ramnit', 'tinba', 'padcrypt', 'vawtrak', 'gozi', 'emotet']

#     csv_file_path = 'dga_classifidga_domains_full.csv'

#     with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
#             csv_reader = csv.reader(csvfile)

#             for family in dga_families:
#                 for row in csv_reader:
#                     if row[1].strip().lower() == family:
#                         domains.append(row[2].strip())  # Adiciona o domínio à lista 'domains'
#                         labels.append(family)  # Adiciona o nome da família à lista 'labels'

#                         # Para garantir que não exceda o número desejado
#                         if len(domains) == num_per_dga:
#                             break
#                 # Volta ao início do arquivo para a próxima família
#                 csvfile.seek(0)

#     return domains, labels

#     #     # Salvando os dados em um arquivo usando pickle
#     #     with open(data_file, 'wb') as file:
#     #         pickle.dump(zip(labels, domains), file)

#     # # Carregando os dados do arquivo
#     # with open(data_file, 'rb') as file:
#     #     return pickle.load(file)


import csv

def get_data(num_per_class=10000):
    domains = []
    labels = []

    csv_file_path = 'dga_classifier/dga_domains_full.csv'

    counts = {'alexa': 0, 'maligno': 0}

    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            label = row[1].strip().lower()
            domain = row[2].strip()

            if label == 'alexa' and counts[label] < num_per_class / 2:
                domains.append(domain)
                labels.append(label)
                counts[label] += 1

            elif label != 'alexa' and counts['maligno'] < num_per_class / 2:
                domains.append(domain)
                labels.append(label)  # Use um rótulo específico para dados malignos
                counts['maligno'] += 1

            if counts['alexa'] == num_per_class / 2 and counts['maligno'] == num_per_class / 2:
                break

                # Salvar os domínios em um arquivo de texto
    with open('dominios_selecionados.txt', 'w', encoding='utf-8') as txtfile:
        for domain, label in zip(domains, labels):
            txtfile.write(f'{domain}: {label}\n')


    return domains, labels

domains, labels = get_data()



