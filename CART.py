from __future__ import annotations
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from typing import Tuple
from abc import ABC,abstractmethod
from scipy.stats import mode
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score

# -------------------------------- CART -------------------------------- #
class Node(object):
    """
    definir e controlar os nós da árvore
    """

    def __init__(self) -> None:
        """
        Inicializador de uma instância de Node
        """
        self.__split = None
        self.__feature = None
        self.__left = None
        self.__right = None
        self.leaf_value = None

    def set_params(self, split: float, feature: int) -> None:
        """
        Define os parâmetros de divisão e característica para este nó

        Entrada:
            split   -> valor para dividir a característica
            feature -> índice da característica a ser usada na divisão
        """
        self.__split = split
        self.__feature = feature

    def get_params(self) -> Tuple[float, int]:
        """
        Obtém os parâmetros de divisão e característica para este nó

        Saída:
            Tupla contendo o par (split, feature)
        """
        return (self.__split, self.__feature)

    def set_children(self, left: Node, right: Node) -> None:
        """
        Define os nós filhos esquerdo/direito para o nó atual

        Entradas:
            left  -> nó filho esquerdo
            right -> nó filho direito
        """
        self.__left = left
        self.__right = right

    def get_left_node(self) -> Node:
        """
        Obtém o nó filho esquerdo

        Saída:
            Nó filho esquerdo
        """
        return self.__left

    def get_right_node(self) -> Node:
        """
        Obtém o nó filho direito

        Saída:
            Nó filho direito
        """
        return self.__right


class DecisionTree(ABC):
    """
    Classe base que abrange o algoritmo CART
    """

    def __init__(self, max_depth: int = None, min_samples_split: int = 2) -> None:
        """
        Inicializador

        Entradas:
            max_depth         -> profundidade máxima que a árvore pode atingir
            min_samples_split -> número mínimo de amostras necessárias para dividir um nó
        """
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    @abstractmethod
    def _impurity(self, D: np.array) -> None:
        """
        Função protegida para definir a impureza
        """
        pass

    @abstractmethod
    def _leaf_value(self, D: np.array) -> None:
        """
        Função protegida para calcular o valor em um nó folha
        """
        pass

    def __grow(self, node: Node, D: np.array, level: int) -> None:
        """
        Função recursiva privada para construir a árvore durante o treinamento

        Entradas:
            node  -> nó atual da árvore
            D     -> amostra de dados no nó
            level -> nível de profundidade na árvore para o nó
        """
        # estamos em um nó folha?
        depth = (self.max_depth is None) or (self.max_depth >= (level + 1))
        msamp = (self.min_samples_split <= D.shape[0])
        n_cls = np.unique(D[:, -1]).shape[0] != 1

        # não é um nó folha
        if depth and msamp and n_cls:

            # inicializa os parâmetros da função
            ip_node = None
            feature = None
            split = None
            left_D = None
            right_D = None
            # itera pelas combinações possíveis de característica/divisão
            for f in range(D.shape[1] - 1):
                for s in np.unique(D[:, f]):
                    # para a combinação atual (f, s), divide o conjunto de dados
                    D_l = D[D[:, f] <= s]
                    D_r = D[D[:, f] > s]
                    # certifica-se de que temos arrays não vazios
                    if D_l.size and D_r.size:
                        # calcula a impureza
                        ip = (D_l.shape[0] / D.shape[0]) * self._impurity(D_l) + (D_r.shape[0] / D.shape[0]) * self._impurity(
                            D_r)
                        # atualiza a impureza e a escolha de (f, s)
                        if (ip_node is None) or (ip < ip_node):
                            ip_node = ip
                            feature = f
                            split = s
                            left_D = D_l
                            right_D = D_r
            # define os parâmetros do nó atual
            node.set_params(split, feature)
            # declara os nós filhos
            left_node = Node()
            right_node = Node()
            node.set_children(left_node, right_node)
            # investiga os nós filhos
            self.__grow(node.get_left_node(), left_D, level + 1)
            self.__grow(node.get_right_node(), right_D, level + 1)

        # é um nó folha
        else:

            # define o valor do nó e retorna
            node.leaf_value = self._leaf_value(D)
            return

    def __traverse(self, node: Node, Xrow: np.array) -> int | float:
        """
        Função recursiva privada para percorrer a árvore treinada

        Entradas:
            node -> nó atual na árvore
            Xrow -> amostra de dados considerada
        Saída:
            valor do nó folha correspondente a Xrow
        """
        # verifica se estamos em um nó folha
        if node.leaf_value is None:
            # obtém os parâmetros no nó
            (s, f) = node.get_params()
            # decide ir para a esquerda ou direita
            if Xrow[f] <= s:
                return self.__traverse(node.get_left_node(), Xrow)
            else:
                return self.__traverse(node.get_right_node(), Xrow)
        else:
            # retorna o valor do nó folha
            return node.leaf_value

    def train(self, Xin: np.array, Yin: np.array) -> None:
        """
        Treina o modelo CART

        Entradas:
            Xin -> conjunto de recursos preditores de entrada
            Yin -> conjunto de rótulos de entrada
        """
        # prepara os dados de entrada
        D = np.concatenate((Xin, Yin.values.reshape(-1, 1)), axis=1)
        # define o nó raiz da árvore
        self.tree = Node()
        # constrói a árvore
        self.__grow(self.tree, D, 1)

    def predict(self, Xin: np.array) -> np.array:
        """
        Faz previsões a partir do modelo CART treinado

        Entrada:
            Xin -> conjunto de recursos preditores de entrada
        Saída:
            array de valores de previsão
        """
        # itera pelas linhas de Xin
        p = []
        for r in range(Xin.shape[0]):
            p.append(self.__traverse(self.tree, Xin[r, :]))
        # retorna as previsões
        return np.array(p).flatten()


class DecisionTreeClassifier(DecisionTree):
    """
    Classificador Árvore de Decisão
    """

    def __init__(self, max_depth: int = None, min_samples_split: int = 2, loss: str = 'gini') -> None:
        """
        Inicializador

        Entradas:
            max_depth         -> profundidade máxima que a árvore pode atingir
            min_samples_split -> número mínimo de amostras necessárias para dividir um nó
            loss              -> função de perda a ser usada durante o treinamento
        """
        DecisionTree.__init__(self, max_depth, min_samples_split)
        self.loss = loss

    def __gini(self, D: np.array) -> float:
        """
        Função privada para definir a impureza de Gini

        Entrada:
            D -> dados para calcular a impureza de Gini
        Saída:
            Impureza de Gini para D
        """
        # inicializa a saída
        G = 0
        # itera pelas classes únicas
        for c in np.unique(D[:, -1]):
            # calcula p para a classe atual c
            p = D[D[:, -1] == c].shape[0] / D.shape[0]
            # calcula o termo para a classe atual c
            G += p * (1 - p)
        # retorna a impureza de Gini
        return G


    def _impurity(self, D: np.array) -> float:
        """
        Função protegida para definir a impureza

        Entrada:
            D -> dados para calcular a métrica de impureza
        Saída:
            Métrica de impureza para D
        """
        # usa a função de perda selecionada para calcular a impureza
        return self.__gini(D)

    def _leaf_value(self, D: np.array) -> float:
        """
        Função protegida para calcular o valor em um nó folha

        Entrada:
            D -> conjunto de dados no nó folha
        Saída:
            Valor do nó folha
        """
        # retorna a classe mais frequente no nó folha
        return mode(D[:, -1], axis=None, keepdims=True)[0][0]

# -------------------------------- pre processamento -------------------------------- #
# Define a função para agrupar as idades
def age_to_group(age):
    if age in range(1, 4):
        return 1
    elif age in range(5, 8):
        return 2
    elif age in range(9, 12):
        return 3
    elif age == 13:
        return 4
    else:
        return 5

# Define a função para agrupar as faixas de renda
def income_to_group(income):
    if income in range(1, 2):
        return 1
    elif income in range(3, 4):
        return 2
    elif income in range(5, 6):
        return 3
    elif income == 7:
        return 4
    else:
        return 5

# Define a função para alterar a logica da classificacao de saude
def genhlth_to_group(genhlth):
    if genhlth == 5:
        return 1
    elif genhlth == 4:
        return 2
    elif genhlth == 3:
        return 3
    elif genhlth == 2:
        return 4
    else:
        return 5


# ler arquivo CSV
datainput = pd.read_csv("bd_diabetes.csv", delimiter=",")

# tratar outliers
for col in datainput.columns:
    if col in ['BMI',  'MentHlth', 'PhysHlth']:
        datainput = datainput[np.abs(
            datainput[col] - datainput[col].mean()) / datainput[col].std() < 3]

datainput = datainput.drop_duplicates()  # eliminar redundancia

# Aplica a função à coluna 'Age' e sobrescreve os valores originais
datainput['Age'] = datainput['Age'].apply(age_to_group)

# Aplica a função à coluna 'Income' e sobrescreve os valores originais
datainput['Income'] = datainput['Income'].apply(income_to_group)

# Aplica a função à coluna 'GenHlth' e sobrescreve os valores originais
datainput['GenHlth'] = datainput['GenHlth'].apply(genhlth_to_group)

# Verificar a matriz de correlação
correlation_matrix = datainput.corr()
# print(correlation_matrix)

# selecionar as colunas de entrada
X = datainput[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
               'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']].values

# selecionar a coluna de saída (rótulo)
y = datainput["Diabetes_binary"]

# undersampling
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)
num_instances = X_resampled.shape[0]
print("Número total de instâncias após a amostragem:", num_instances)
counts = np.bincount(y_resampled)
total_classe_0 = counts[0]
total_classe_1 = counts[1]
print("Total da classe 0:", total_classe_0)
print("Total da classe 1:", total_classe_1)


kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Inicializa as listas para armazenar as métricas em cada fold
accuracies = []
precisions = []
recalls = []
f1s = []
precisions_class0 = []
recalls_class0 = []
f1s_class0 = []
precisions_class1 = []
recalls_class1 = []
f1s_class1 = []

# Itera sobre cada fold
for train_index, test_index in kf.split(X_resampled):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    clf = DecisionTreeClassifier(max_depth=5,loss='gini')
    clf.train(X_train,y_train)
    # generate predictions
    previsoes = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, previsoes))
    precisions.append(precision_score(y_test, previsoes, average='macro'))
    recalls.append(recall_score(y_test, previsoes, average='macro'))
    f1s.append(f1_score(y_test, previsoes, average='macro'))

    # armazena as métricas deste fold para classe 0
    precisions_class0.append(precision_score(y_test, previsoes, pos_label=0))
    recalls_class0.append(recall_score(y_test, previsoes, pos_label=0))
    f1s_class0.append(f1_score(y_test, previsoes, pos_label=0))

    # armazena as métricas deste fold para classe 1
    precisions_class1.append(precision_score(y_test, previsoes, pos_label=1))
    recalls_class1.append(recall_score(y_test, previsoes, pos_label=1))
    f1s_class1.append(f1_score(y_test, previsoes, pos_label=1))

print("\nResultados gerais:")
print("Acurácia Média:", np.mean(accuracies))
print("Precisão Média:", np.mean(precisions))
print("Recall Médio:", np.mean(recalls))
print("F1-Score Médio:", np.mean(f1s))
        
print("\nResultados classe 0:")
print("Precisão Classe 0 Média:", np.mean(precisions_class0))
print("Recall Classe 0 Médio:", np.mean(recalls_class0))
print("F1-Score Classe 0 Médio:", np.mean(f1s_class0))
        
print("\nResultados classe 1:")
print("Precisão Classe 1 Média:", np.mean(precisions_class1))
print("Recall Classe 1 Médio:", np.mean(recalls_class1))
print("F1-Score Classe 1 Médio:", np.mean(f1s_class1))