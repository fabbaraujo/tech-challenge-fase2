"""Testes do módulo de LLM. Nenhum teste chama API: as respostas são escritas à mão."""

import json

import pytest

from src import llm


@pytest.fixture
def caso():
    fatores = [
        {"variavel": "Follicle No. (R)", "descricao": "nº de folículos, ovário direito", "valor": 14, "shap": 1.3},
        {"variavel": "AMH(ng/mL)", "descricao": "hormônio antimülleriano", "valor": 8.2, "shap": 0.7},
        {"variavel": "Cycle(R/I)", "descricao": "ciclo regular (2) ou irregular (4)", "valor": 4, "shap": 0.5},
        {"variavel": "Weight gain(Y/N)", "descricao": "ganho de peso", "valor": 1, "shap": 0.3},
        {"variavel": "Age (yrs)", "descricao": "idade", "valor": 31, "shap": -0.2},
    ]
    return llm.montar_caso(0.82, 0.5, fatores)


NOMES = ["Follicle No. (R)", "Follicle No. (L)", "AMH(ng/mL)", "Cycle(R/I)", "Weight gain(Y/N)",
         "Age (yrs)", "TSH (mIU/L)", "BMI", "PRL(ng/mL)"]


# --- montagem do caso e dos prompts ---------------------------------------

def test_montar_caso_define_decisao_e_efeito(caso):
    assert caso["decisao"] == "SOP provável"
    assert caso["fatores"][0]["efeito"] == "aumenta"
    assert caso["fatores"][-1]["efeito"] == "reduz"


@pytest.mark.parametrize("versao", llm.VERSOES)
def test_prompt_contem_probabilidade_e_os_cinco_fatores(caso, versao):
    system, user = llm.montar_prompt(versao, caso)
    real = user.split("Agora o caso real.")[-1]
    assert "82%" in real
    for f in caso["fatores"]:
        assert f["variavel"] in real


@pytest.mark.parametrize("versao", llm.VERSOES)
def test_prompt_nao_tem_identificador_de_paciente(caso, versao):
    system, user = llm.montar_prompt(versao, caso)
    texto = (system + user).lower()
    for proibido in ("patient file", "sl. no", "id_caso", "prontuário", "nome da paciente"):
        assert proibido not in texto


def test_v1_e_ingenua_e_v2_tem_regras(caso):
    s1, _ = llm.montar_prompt("v1", caso)
    s2, _ = llm.montar_prompt("v2", caso)
    assert s1 == ""
    assert "NÃO faz diagnóstico" in s2 and "SOMENTE os dados fornecidos" in s2


def test_v3_pede_json_e_traz_dois_exemplos(caso):
    system, user = llm.montar_prompt("v3", caso)
    assert "JSON" in system                      # o modo JSON da API exige a palavra nas instruções
    assert user.count("Exemplo ") == 2
    for ex in llm.EXEMPLOS_V3:                   # os exemplos são, eles mesmos, saídas válidas
        assert llm.extrair_json(json.dumps(ex["saida"], ensure_ascii=False))[1] is None


def test_contexto_textual_entra_no_prompt(caso):
    _, user = llm.montar_prompt("v2", caso, contexto_textual="queixa de acne há 2 anos")
    assert "queixa de acne há 2 anos" in user


def test_versao_desconhecida(caso):
    with pytest.raises(ValueError):
        llm.montar_prompt("v9", caso)


# --- extração do JSON -----------------------------------------------------

BOA_V3 = json.dumps({
    "resumo": "O modelo estima 82% de probabilidade de SOP.",
    "fatores_principais": ["Follicle No. (R)", "AMH(ng/mL)", "Cycle(R/I)", "Weight gain(Y/N)", "Age (yrs)"],
    "confianca": "alta",
    "sugestao_investigacao": "Confirmar com a avaliação hormonal.",
    "limitacoes": "Não substitui a avaliação médica."}, ensure_ascii=False)


@pytest.mark.parametrize("texto,valido", [
    (BOA_V3, True),
    ("```json\n" + BOA_V3 + "\n```", True),
    ('{"resumo": "x", "confianca": "alta"}', False),
    ('{"resumo": "x", "fatores_principais": "AMH", "confianca": "a", "sugestao_investigacao": "b", "limitacoes": "c"}', False),
    ("Claro! A paciente provavelmente tem SOP.", False),
    ("[1, 2, 3]", False),
    (None, False),
])
def test_extrair_json(texto, valido):
    assert (llm.extrair_json(texto)[1] is None) == valido


# --- checagens: uma resposta boa e uma ruim para cada uma ------------------

BOA_V2 = ('O modelo estima 82% de probabilidade de SOP. Os fatores que mais pesaram foram "Follicle No. (R)", '
          '"AMH(ng/mL)", "Cycle(R/I)" e "Weight gain(Y/N)", que aumentam a chance, e "Age (yrs)", que reduz. '
          "Esta análise não substitui a avaliação médica.")


def test_resposta_boa_passa_em_tudo(caso):
    c = llm.checar_resposta(BOA_V2, caso, "v2", NOMES)
    assert c["sem_fator_inventado"] and c["probabilidade_correta"] and c["tem_ressalva"]
    assert not c["diagnostico_definitivo"]
    assert c["cobertura_fatores"] == 1.0


def test_detecta_fator_inventado_no_texto(caso):
    ruim = BOA_V2 + ' O "TSH (mIU/L)" também está alterado.'
    c = llm.checar_resposta(ruim, caso, "v2", NOMES)
    assert c["fatores_inventados"] == ["TSH (mIU/L)"]
    assert not c["sem_fator_inventado"]


def test_detecta_fator_inventado_na_lista_da_v3(caso):
    dados = json.loads(BOA_V3)
    dados["fatores_principais"].append("Insulina de jejum")
    c = llm.checar_resposta(json.dumps(dados, ensure_ascii=False), caso, "v3", NOMES)
    assert c["fatores_inventados"] == ["Insulina de jejum"]


def test_comparacao_de_fator_ignora_maiusculas_e_acentos(caso):
    texto = BOA_V2.replace("Follicle No. (R)", "follicle no. (r)")
    assert llm.checar_resposta(texto, caso, "v2", NOMES)["sem_fator_inventado"]


def test_detecta_probabilidade_errada(caso):
    c = llm.checar_resposta(BOA_V2.replace("82%", "95%"), caso, "v2", NOMES)
    assert c["cita_probabilidade"] and not c["probabilidade_correta"]


def test_aceita_probabilidade_com_casa_decimal(caso):
    assert llm.checar_resposta(BOA_V2.replace("82%", "82,0 %"), caso, "v2", NOMES)["probabilidade_correta"]


@pytest.mark.parametrize("frase", ["A paciente tem SOP.", "Diagnóstico confirmado.",
                                   "A paciente está com síndrome dos ovários policísticos.",
                                   "A paciente não tem SOP."])
def test_detecta_diagnostico_definitivo(caso, frase):
    assert llm.checar_resposta(BOA_V2 + " " + frase, caso, "v2", NOMES)["diagnostico_definitivo"]


def test_linguagem_de_probabilidade_nao_e_diagnostico(caso):
    texto = "O modelo sugere SOP provável, com 82% de probabilidade. Não substitui a avaliação médica."
    assert not llm.checar_resposta(texto, caso, "v2", NOMES)["diagnostico_definitivo"]


def test_detecta_falta_de_ressalva(caso):
    sem = BOA_V2.replace("Esta análise não substitui a avaliação médica.", "")
    assert not llm.checar_resposta(sem, caso, "v2", NOMES)["tem_ressalva"]


def test_v3_valida_e_invalida(caso):
    assert llm.checar_resposta(BOA_V3, caso, "v3", NOMES)["json_valido"]
    assert not llm.checar_resposta("texto solto, 82%", caso, "v3", NOMES)["json_valido"]


def test_resposta_vazia(caso):
    assert llm.checar_resposta(None, caso, "v2", NOMES) == {"respondeu": False}


# --- cliente ---------------------------------------------------------------

def test_sem_chave_o_cliente_fica_offline(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm.provedor_ativo() == "offline"
    r = llm.gerar("sistema", "usuário")
    assert r["provedor"] == "offline" and r["texto"] is None and r["erro"] is None


def test_provedor_desconhecido_vira_offline(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "inexistente")
    assert llm.provedor_ativo() == "offline"


def test_nova_tentativa_em_erro_repetivel():
    class RateLimitError(Exception):
        pass

    falhas = iter([True, True, False])

    def instavel():
        if next(falhas):
            raise RateLimitError("429")
        return "ok"

    assert llm.com_novas_tentativas(instavel, espera_inicial=0.01) == "ok"


def test_erro_nao_repetivel_falha_na_hora():
    class AuthenticationError(Exception):
        pass

    chamadas = []

    def chave_errada():
        chamadas.append(1)
        raise AuthenticationError("401")

    with pytest.raises(AuthenticationError):
        llm.com_novas_tentativas(chave_errada, espera_inicial=0.01)
    assert len(chamadas) == 1


def test_registro_de_interacoes(tmp_path):
    arq = tmp_path / "interacoes.jsonl"
    llm.registrar_interacao(arq, {"id_caso": 1, "versao": "v2", "texto": "ok"})
    llm.registrar_interacao(arq, {"id_caso": 1, "versao": "v3", "texto": "{}"})
    assert [r["versao"] for r in llm.ler_interacoes(arq)] == ["v2", "v3"]
    assert llm.ler_interacoes(tmp_path / "nao_existe.jsonl") == []
