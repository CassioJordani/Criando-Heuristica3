import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_digits

# carregar os dados
digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target

def heuristica_par(tamanho_df):
    predicoes = []
    acertos_adicionais = 0
    
    # Loop de 0 a 9 que se repete usando o resto da divisão (%)
    for i in range(tamanho_df):
        num = i % 10
        predicoes.append(num)
        
        # Verifica se é divisível por 2
        if num % 2 == 0:
            acertos_adicionais += 1
            
    return predicoes, acertos_adicionais

# Executando a nova função criada
lista_predicoes, acertos_par = heuristica_par(len(df))
df['predicao'] = lista_predicoes

# Cálculo de acertos, sendo o real alvo + bônus por ser par
acertos_reais = (df['target'] == df['predicao']).sum()
total_acertos_final = acertos_reais + acertos_par
acuracia_final = total_acertos_final / len(df)
acuracia = acertos_reais / len(df)

print(f'Total de registros: {len(df)}')
print(f'\n')
print(f'Acertos reais: {acertos_reais}')
print(f'Acertos por números pares: {acertos_par}')
print(f'Total de acertos combinados: {total_acertos_final}')
print(f'\n')
print(f'Acurácia: {acuracia:.4f}')
print(f'Acurácia combinada: {acuracia_final:.4f}')

# Mostrando a matriz da imagem
idx = random.randint(0, len(digits.images) - 1)
plt.gray()
plt.matshow(digits.images[idx])
plt.title(f"Real: {digits.target[idx]} | Predição: {df['predicao'].iloc[idx]}")
plt.show()
