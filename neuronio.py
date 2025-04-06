import random

# Dataset
dados = [
    [-1, -0.6508, 0.1097, 4.0009, -1],
    [-1, -1.4492, 0.8896, 4.4005, -1],
    [-1,  2.0850, 0.6876, 12.0710, -1],
    [-1,  0.2626, 1.1476, 7.7985, 1],
    [-1,  0.6418, 1.0234, 7.0427, 1],
    [-1,  0.2569, 0.6730, 8.3265, -1],
    [-1,  1.1155, 0.6043, 7.4446, 1],
    [-1,  0.0914, 0.3399, 7.0677, -1],
    [-1,  0.0121, 0.5256, 4.6316, 1],
    [-1, -0.0429, 0.4660, 5.4323, 1],
    [-1,  0.4340, 0.6870, 8.2287, -1],
    [-1,  0.2735, 1.0287, 7.1934, 1],
    [-1,  0.4839, 0.4851, 7.4850, -1],
    [-1,  0.4089, -0.1267, 5.5019, -1],
    [-1,  1.4391, 0.1614, 8.5843, -1],
    [-1, -0.9115, -0.1973, 2.1962, -1],
    [-1,  0.3654, 1.0475, 7.4858, 1],
    [-1,  0.2144, 0.7515, 7.1699, 1],
    [-1,  0.2013, 1.0014, 6.5489, 1],
    [-1,  0.6483, 0.2183, 5.8991, 1],
    [-1, -0.1147, 0.2242, 7.2435, -1],
    [-1, -0.7970, 0.8795, 3.8762, 1],
    [-1, -1.0625, 0.6366, 2.4707, 1],
    [-1,  0.5307, 0.1285, 5.6883, 1],
    [-1, -1.2200, 0.7777, 1.7252, 1],
    [-1,  0.3957, 0.1076, 5.6623, -1],
    [-1, -0.1013, 0.5989, 7.1812, -1],
    [-1,  2.4482, 0.9455, 11.2095, 1],
    [-1,  2.0149, 0.6192, 10.9263, -1],
    [-1,  0.2012, 0.2611, 5.4631, 1],
]

# Embaralhar os dados
random.shuffle(dados)

# Separar entrada e saída
X = [linha[:-1] for linha in dados]
Y = [linha[-1] for linha in dados]

# Separar 2/3 treino e 1/3 teste
tamanho_treino = int(2/3 * len(X))
X_treino, Y_treino = X[:tamanho_treino], Y[:tamanho_treino]
X_teste, Y_teste = X[tamanho_treino:], Y[tamanho_treino:]

def degrau_bipolar(u):
    return 1 if u > 0 else -1 if u < 0 else 0

def treino_neuronio(X, Yd, taxa_aprendizado=0.1, max_epocas=1000):
    pesos = [random.uniform(-1, 1) for _ in range(len(X[0]))]
    for epoca in range(max_epocas):
        erro_total = 0
        for i in range(len(X)):
            u = sum([x*w for x, w in zip(X[i], pesos)])
            Yp = degrau_bipolar(u)
            erro = Yd[i] - Yp
            pesos = [w + taxa_aprendizado * erro * x for w, x in zip(pesos, X[i])]
            erro_total += abs(erro)
        if erro_total == 0:
            break
    return pesos

def testar_neuronio(X, Y, pesos):
    acertos = 0
    for i in range(len(X)):
        u = sum([x*w for x, w in zip(X[i], pesos)])
        Yp = degrau_bipolar(u)
        if Yp == Y[i]:
            acertos += 1
    return acertos / len(Y), acertos, len(Y)

# Treinar e testar
pesos_finais = treino_neuronio(X_treino, Y_treino)
acuracia, acertos, total = testar_neuronio(X_teste, Y_teste, pesos_finais)

# Resultados
print(f"\nPesos finais: {pesos_finais}")
print(f"Acurácia no teste: {acuracia:.2%} ({acertos}/{total} acertos)")
