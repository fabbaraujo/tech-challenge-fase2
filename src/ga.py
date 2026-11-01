"""Algoritmo genético para busca de hiperparâmetros.

Este módulo não conhece o scikit-learn. Ele recebe uma função
``avaliar(params) -> float`` e devolve o melhor conjunto de parâmetros
encontrado. A função de fitness (pipeline + validação cruzada + F2) fica no
notebook. Assim o mesmo AG serve para os três modelos e os testes rodam com
uma fitness de brinquedo, em milissegundos.

Codificação híbrida (aula 3 de AG): cada gene tem um tipo.

- ``("real", lo, hi)``      número real em [lo, hi]
- ``("log10", lo, hi)``     real em [lo, hi]; o valor decodificado é 10 ** gene
- ``("inteiro", lo, hi)``   inteiro em [lo, hi]
- ``("categorico", [v1, v2, ...])``  o gene guarda o índice do valor escolhido
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Espaços de busca (seção 4.3 do overview)
# ---------------------------------------------------------------------------

ESPACOS = {
    "lr": {
        "C": ("log10", -3.0, 2.0),                       # C de 0,001 a 100
        "l1_ratio": ("categorico", [0.0, 1.0]),          # 0 = L2, 1 = L1 (sklearn 1.8+)
        "class_weight": ("categorico", [None, "balanced"]),
    },
    "rf": {
        "n_estimators": ("inteiro", 50, 500),
        "max_depth": ("inteiro", 2, 20),
        "min_samples_split": ("inteiro", 2, 20),
        "max_features": ("real", 0.1, 1.0),
        "class_weight": ("categorico", [None, "balanced"]),
    },
    "knn": {
        "n_neighbors": ("inteiro", 1, 50),
        "weights": ("categorico", ["uniform", "distance"]),
        "p": ("categorico", [1, 2]),                     # 1 = Manhattan, 2 = Euclidiana
    },
}

CASAS_DECIMAIS = 3  # genes reais são arredondados na decodificação (ajuda o cache)


# ---------------------------------------------------------------------------
# Codificação
# ---------------------------------------------------------------------------

def individuo_aleatorio(espaco: dict, rng: np.random.Generator) -> dict:
    """Sorteia um indivíduo dentro das faixas do espaço."""
    ind = {}
    for gene, (tipo, *args) in espaco.items():
        if tipo in ("real", "log10"):
            ind[gene] = float(rng.uniform(args[0], args[1]))
        elif tipo == "inteiro":
            ind[gene] = int(rng.integers(args[0], args[1] + 1))
        elif tipo == "categorico":
            ind[gene] = int(rng.integers(len(args[0])))
        else:
            raise ValueError(f"tipo de gene desconhecido: {tipo}")
    return ind


def decodificar(espaco: dict, ind: dict) -> dict:
    """Transforma o indivíduo no dicionário de hiperparâmetros do modelo."""
    params = {}
    for gene, (tipo, *args) in espaco.items():
        valor = ind[gene]
        if tipo == "real":
            params[gene] = round(float(valor), CASAS_DECIMAIS)
        elif tipo == "log10":
            params[gene] = float(10 ** round(float(valor), CASAS_DECIMAIS))
        elif tipo == "inteiro":
            params[gene] = int(valor)
        elif tipo == "categorico":
            params[gene] = args[0][int(valor)]
    return params


def chave(espaco: dict, ind: dict) -> tuple:
    """Identidade do indivíduo para o cache: os parâmetros decodificados."""
    return tuple(sorted((k, str(v)) for k, v in decodificar(espaco, ind).items()))


# ---------------------------------------------------------------------------
# Operadores (aulas 3, 4 e 5 de AG)
# ---------------------------------------------------------------------------

def selecao_torneio(notas, k: int, rng: np.random.Generator) -> int:
    """Sorteia k indivíduos e devolve o índice do melhor deles."""
    competidores = rng.choice(len(notas), size=min(k, len(notas)), replace=False)
    return int(max(competidores, key=lambda i: notas[i]))


def selecao_roleta(notas, rng: np.random.Generator) -> int:
    """Probabilidade proporcional à fitness (deslocada para ficar positiva)."""
    notas = np.asarray(notas, dtype=float)
    pesos = notas - notas.min() + 1e-9
    return int(rng.choice(len(notas), p=pesos / pesos.sum()))


def cruzamento(espaco: dict, pai1: dict, pai2: dict, rng: np.random.Generator) -> dict:
    """Crossover por tipo de gene: aritmético para números, uniforme para categorias."""
    filho = {}
    for gene, (tipo, *args) in espaco.items():
        if tipo in ("real", "log10", "inteiro"):
            alfa = rng.random()
            valor = alfa * pai1[gene] + (1 - alfa) * pai2[gene]
            filho[gene] = int(round(valor)) if tipo == "inteiro" else float(valor)
        else:
            filho[gene] = pai1[gene] if rng.random() < 0.5 else pai2[gene]
    return filho


def mutacao(espaco: dict, ind: dict, taxa: float, rng: np.random.Generator,
            escala: float = 0.1) -> dict:
    """Cada gene muta com probabilidade ``taxa``.

    Números: ruído gaussiano com desvio de ``escala`` vezes a largura da faixa,
    e ``clip`` para não sair dela. Categorias: troca por outro valor.
    """
    novo = dict(ind)
    for gene, (tipo, *args) in espaco.items():
        if rng.random() >= taxa:
            continue
        if tipo in ("real", "log10", "inteiro"):
            lo, hi = args
            valor = ind[gene] + rng.normal(0, escala * (hi - lo))
            valor = float(np.clip(valor, lo, hi))
            novo[gene] = int(round(valor)) if tipo == "inteiro" else valor
        else:
            opcoes = [i for i in range(len(args[0])) if i != ind[gene]]
            if opcoes:
                novo[gene] = int(rng.choice(opcoes))
    return novo


# ---------------------------------------------------------------------------
# O loop
# ---------------------------------------------------------------------------

@dataclass
class ResultadoAG:
    melhores_params: dict
    melhor_fitness: float
    historico: list = field(default_factory=list)
    avaliacoes: int = 0          # fitness calculadas de fato
    consultas: int = 0           # pedidos de fitness, incluindo os atendidos pelo cache

    @property
    def economia_cache(self) -> float:
        return 1 - self.avaliacoes / self.consultas if self.consultas else 0.0


def executar_ag(espaco: dict, avaliar, tam_pop: int = 20, geracoes: int = 15,
                taxa_mutacao: float = 0.2, taxa_cruzamento: float = 0.9,
                selecao: str = "torneio", k_torneio: int = 3, elitismo: int = 2,
                semente: int = 42, rotulo: str = "") -> ResultadoAG:
    """Roda o AG e devolve o melhor indivíduo e o histórico por geração.

    ``avaliar`` recebe o dicionário de hiperparâmetros e devolve a fitness
    (quanto maior, melhor).
    """
    if selecao not in ("torneio", "roleta"):
        raise ValueError("selecao deve ser 'torneio' ou 'roleta'")
    rng = np.random.default_rng(semente)
    cache: dict = {}
    consultas = 0

    def fitness(ind):
        nonlocal consultas
        consultas += 1
        k = chave(espaco, ind)
        if k not in cache:
            cache[k] = float(avaliar(decodificar(espaco, ind)))
        return cache[k]

    def escolher(notas):
        if selecao == "torneio":
            return selecao_torneio(notas, k_torneio, rng)
        return selecao_roleta(notas, rng)

    populacao = [individuo_aleatorio(espaco, rng) for _ in range(tam_pop)]
    historico = []
    inicio = time.perf_counter()

    for g in range(geracoes):
        t0 = time.perf_counter()
        notas = [fitness(ind) for ind in populacao]
        ordem = list(np.argsort(notas)[::-1])
        unicos = len({chave(espaco, ind) for ind in populacao})
        registro = {
            "geracao": g,
            "melhor": notas[ordem[0]],
            "media": float(np.mean(notas)),
            "desvio": float(np.std(notas)),
            "unicos": unicos,
            "avaliacoes": len(cache),
            "consultas": consultas,
            "tempo_s": round(time.perf_counter() - t0, 3),
            "melhor_params": decodificar(espaco, populacao[ordem[0]]),
        }
        historico.append(registro)
        logger.info("%s geracao=%d melhor=%.4f media=%.4f desvio=%.4f unicos=%d avaliacoes=%d tempo_s=%.2f",
                    rotulo, g, registro["melhor"], registro["media"], registro["desvio"],
                    unicos, len(cache), registro["tempo_s"])

        if g == geracoes - 1:
            break

        nova = [populacao[i] for i in ordem[:elitismo]]            # elitismo
        while len(nova) < tam_pop:
            pai1 = populacao[escolher(notas)]
            pai2 = populacao[escolher(notas)]
            filho = cruzamento(espaco, pai1, pai2, rng) if rng.random() < taxa_cruzamento else dict(pai1)
            nova.append(mutacao(espaco, filho, taxa_mutacao, rng))
        populacao = nova

    melhor = populacao[ordem[0]]
    logger.info("%s fim melhor=%.4f params=%s avaliacoes=%d consultas=%d tempo_total_s=%.1f",
                rotulo, notas[ordem[0]], decodificar(espaco, melhor), len(cache), consultas,
                time.perf_counter() - inicio)
    return ResultadoAG(melhores_params=decodificar(espaco, melhor), melhor_fitness=notas[ordem[0]],
                       historico=historico, avaliacoes=len(cache), consultas=consultas)
