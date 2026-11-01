"""Testes do algoritmo genético.

O AG é estocástico e o ótimo real é desconhecido, então os testes verificam
propriedades que precisam valer sempre, com uma fitness de brinquedo de ótimo
conhecido. Nenhum teste treina modelo: tudo roda em poucos segundos.
"""

import numpy as np
import pytest

from src import ga


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def fitness_brinquedo(params):
    """Máximo em C = 10 (log10 C = 1), com bônus para class_weight balanced."""
    return -(np.log10(params["C"]) - 1) ** 2 + (0.1 if params["class_weight"] == "balanced" else 0)


def _valido(espaco, params):
    for gene, (tipo, *args) in espaco.items():
        v = params[gene]
        if tipo == "real":
            assert args[0] <= v <= args[1], gene
        elif tipo == "log10":
            assert 10 ** args[0] - 1e-9 <= v <= 10 ** args[1] + 1e-9, gene
        elif tipo == "inteiro":
            assert isinstance(v, int) and args[0] <= v <= args[1], gene
        else:
            assert v in args[0], gene


# --- codificação -----------------------------------------------------------

@pytest.mark.parametrize("modelo", list(ga.ESPACOS))
def test_individuos_aleatorios_decodificam_para_parametros_validos(modelo, rng):
    espaco = ga.ESPACOS[modelo]
    for _ in range(1000):
        _valido(espaco, ga.decodificar(espaco, ga.individuo_aleatorio(espaco, rng)))


def test_c_da_regressao_logistica_e_codificado_pelo_expoente():
    espaco = ga.ESPACOS["lr"]
    params = ga.decodificar(espaco, {"C": 1.0, "l1_ratio": 1, "class_weight": 0})
    assert params["C"] == pytest.approx(10.0)
    assert params["l1_ratio"] == 1.0
    assert params["class_weight"] is None


def test_genes_inteiros_saem_inteiros(rng):
    espaco = ga.ESPACOS["rf"]
    params = ga.decodificar(espaco, ga.individuo_aleatorio(espaco, rng))
    for gene in ("n_estimators", "max_depth", "min_samples_split"):
        assert isinstance(params[gene], int)


# --- operadores ------------------------------------------------------------

@pytest.mark.parametrize("modelo", list(ga.ESPACOS))
@pytest.mark.parametrize("taxa", [0.0, 0.5, 1.0])
def test_mutacao_nunca_sai_da_faixa(modelo, taxa, rng):
    espaco = ga.ESPACOS[modelo]
    ind = ga.individuo_aleatorio(espaco, rng)
    for _ in range(300):
        ind = ga.mutacao(espaco, ind, taxa, rng, escala=0.5)   # escala alta força o clip
        _valido(espaco, ga.decodificar(espaco, ind))


def test_mutacao_com_taxa_zero_nao_muda_nada(rng):
    espaco = ga.ESPACOS["rf"]
    ind = ga.individuo_aleatorio(espaco, rng)
    assert ga.mutacao(espaco, ind, 0.0, rng) == ind


def test_mutacao_categorica_troca_o_valor(rng):
    espaco = ga.ESPACOS["knn"]
    ind = ga.individuo_aleatorio(espaco, rng)
    mutado = ga.mutacao(espaco, ind, 1.0, rng)
    assert mutado["weights"] != ind["weights"] and mutado["p"] != ind["p"]


@pytest.mark.parametrize("modelo", list(ga.ESPACOS))
def test_cruzamento_fica_entre_os_pais_e_dentro_da_faixa(modelo, rng):
    espaco = ga.ESPACOS[modelo]
    for _ in range(300):
        p1, p2 = ga.individuo_aleatorio(espaco, rng), ga.individuo_aleatorio(espaco, rng)
        filho = ga.cruzamento(espaco, p1, p2, rng)
        _valido(espaco, ga.decodificar(espaco, filho))
        for gene, (tipo, *_) in espaco.items():
            if tipo == "categorico":
                assert filho[gene] in (p1[gene], p2[gene])
            else:
                assert min(p1[gene], p2[gene]) - 0.5 <= filho[gene] <= max(p1[gene], p2[gene]) + 0.5


def test_torneio_com_k_igual_a_populacao_escolhe_o_melhor(rng):
    notas = [0.1, 0.9, 0.5, 0.3]
    assert all(ga.selecao_torneio(notas, k=4, rng=rng) == 1 for _ in range(20))


def test_roleta_favorece_quem_tem_fitness_maior(rng):
    notas = [0.0, 0.0, 1.0]
    escolhas = [ga.selecao_roleta(notas, rng) for _ in range(200)]
    assert escolhas.count(2) > 190


# --- loop ------------------------------------------------------------------

def test_mesma_semente_mesmo_resultado():
    a = ga.executar_ag(ga.ESPACOS["lr"], fitness_brinquedo, tam_pop=10, geracoes=5, semente=7)
    b = ga.executar_ag(ga.ESPACOS["lr"], fitness_brinquedo, tam_pop=10, geracoes=5, semente=7)
    assert a.melhores_params == b.melhores_params
    assert [h["melhor"] for h in a.historico] == [h["melhor"] for h in b.historico]


@pytest.mark.parametrize("selecao", ["torneio", "roleta"])
def test_elitismo_nao_deixa_o_melhor_piorar(selecao):
    res = ga.executar_ag(ga.ESPACOS["lr"], fitness_brinquedo, tam_pop=10, geracoes=15, selecao=selecao)
    melhores = [h["melhor"] for h in res.historico]
    assert all(depois >= antes for antes, depois in zip(melhores, melhores[1:]))


def test_cache_evita_reavaliacao():
    chamadas = []

    def contando(params):
        chamadas.append(params)
        return fitness_brinquedo(params)

    res = ga.executar_ag(ga.ESPACOS["lr"], contando, tam_pop=10, geracoes=6, elitismo=2)
    assert len(chamadas) == res.avaliacoes
    assert res.consultas == 10 * 6
    assert res.avaliacoes < res.consultas           # a elite volta toda geração e não é recalculada
    assert 0 < res.economia_cache < 1


def test_encontra_o_otimo_da_fitness_de_brinquedo():
    res = ga.executar_ag(ga.ESPACOS["lr"], fitness_brinquedo, tam_pop=20, geracoes=20)
    assert np.log10(res.melhores_params["C"]) == pytest.approx(1.0, abs=0.2)
    assert res.melhores_params["class_weight"] == "balanced"


def test_historico_tem_as_metricas_de_convergencia_e_diversidade():
    res = ga.executar_ag(ga.ESPACOS["knn"], lambda p: -p["n_neighbors"], tam_pop=6, geracoes=3)
    assert len(res.historico) == 3
    for chave in ("melhor", "media", "desvio", "unicos", "avaliacoes", "tempo_s", "melhor_params"):
        assert chave in res.historico[0]


def test_selecao_desconhecida_gera_erro():
    with pytest.raises(ValueError):
        ga.executar_ag(ga.ESPACOS["lr"], fitness_brinquedo, selecao="ranking")
