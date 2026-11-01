"""Integração com LLM: cliente com troca de provedor, prompts e checagens.

Três partes, todas sem dependência do scikit-learn:

1. ``gerar``: chama o provedor escolhido em ``LLM_PROVIDER`` (groq, openai,
   gemini). Sem chave configurada, cai no modo offline e não quebra.
2. ``montar_prompt``: as três versões de prompt (v1 ingênua, v2 com papel e
   regras, v3 com exemplos e saída em JSON).
3. ``checar_resposta``: checagens automáticas de fidelidade da resposta
   (fator inventado, número errado, diagnóstico definitivo, ressalva, JSON).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

MODELO_PADRAO = "openai/gpt-oss-120b"
CHAVES = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
VERSOES = ("v1", "v2", "v3")
CAMPOS_V3 = {"resumo": str, "fatores_principais": list, "confianca": str,
             "sugestao_investigacao": str, "limitacoes": str}


# ---------------------------------------------------------------------------
# 1. Cliente
# ---------------------------------------------------------------------------

def _mensagens(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _groq(system, user, modelo, temperatura, json_mode):
    from groq import Groq
    extra = {"reasoning_effort": "low"} if "gpt-oss" in modelo else {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    r = Groq(max_retries=0).chat.completions.create(
        model=modelo, temperature=temperatura, messages=_mensagens(system, user), **extra)
    return {"texto": r.choices[0].message.content,
            "tokens_entrada": r.usage.prompt_tokens, "tokens_saida": r.usage.completion_tokens}


def _openai(system, user, modelo, temperatura, json_mode):
    from openai import OpenAI
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = OpenAI(max_retries=0).chat.completions.create(
        model=modelo, temperature=temperatura, messages=_mensagens(system, user), **extra)
    return {"texto": r.choices[0].message.content,
            "tokens_entrada": r.usage.prompt_tokens, "tokens_saida": r.usage.completion_tokens}


def _gemini(system, user, modelo, temperatura, json_mode):
    from google import genai
    from google.genai import types
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperatura,
                                      response_mime_type="application/json" if json_mode else None)
    r = genai.Client().models.generate_content(model=modelo, contents=user, config=cfg)
    return {"texto": r.text, "tokens_entrada": r.usage_metadata.prompt_token_count,
            "tokens_saida": r.usage_metadata.candidates_token_count}


def _offline(system, user, modelo, temperatura, json_mode):
    return {"texto": None, "tokens_entrada": 0, "tokens_saida": 0}


PROVEDORES = {"groq": _groq, "openai": _openai, "gemini": _gemini, "offline": _offline}


def provedor_ativo() -> str:
    """O provedor de ``LLM_PROVIDER``, ou ``offline`` se a chave dele não existir."""
    p = os.getenv("LLM_PROVIDER", "groq").lower()
    if p not in CHAVES:
        return "offline"
    return p if os.getenv(CHAVES[p]) else "offline"


def _erro_repetivel(e: Exception) -> bool:
    """Limite de taxa, timeout e falha de conexão valem nova tentativa. Chave errada não."""
    nome = type(e).__name__
    return nome in {"RateLimitError", "APITimeoutError", "APIConnectionError",
                    "InternalServerError", "ResourceExhausted", "ServiceUnavailable"}


def com_novas_tentativas(funcao, tentativas: int = 4, espera_inicial: float = 2.0,
                         repetivel=_erro_repetivel):
    """Chama ``funcao``; em erro repetível espera 2 s, 4 s, 8 s... e tenta de novo."""
    for i in range(tentativas):
        try:
            return funcao()
        except Exception as e:
            if not repetivel(e) or i == tentativas - 1:
                raise
            espera = espera_inicial * 2 ** i
            logger.warning("tentativa=%d erro=%s esperando_s=%.1f", i + 1, type(e).__name__, espera)
            time.sleep(espera)


def gerar(system: str, user: str, temperatura: float = 0.0, json_mode: bool = False,
          provedor: str | None = None, modelo: str | None = None) -> dict:
    """Interface única. Devolve texto, tokens, latência, provedor, modelo e erro."""
    provedor = provedor or provedor_ativo()
    modelo = modelo or os.getenv("LLM_MODEL", MODELO_PADRAO)
    inicio = time.perf_counter()
    try:
        r = com_novas_tentativas(lambda: PROVEDORES[provedor](system, user, modelo, temperatura, json_mode))
        erro = None
    except Exception as e:  # o erro é registrado, o notebook segue
        r, erro = {"texto": None, "tokens_entrada": 0, "tokens_saida": 0}, f"{type(e).__name__}: {e}"
    r.update(provedor=provedor, modelo=modelo, erro=erro,
             latencia_s=round(time.perf_counter() - inicio, 3))
    logger.info("provedor=%s modelo=%s latencia_s=%.3f tokens_in=%s tokens_out=%s erro=%s",
                provedor, modelo, r["latencia_s"], r["tokens_entrada"], r["tokens_saida"], erro)
    return r


# ---------------------------------------------------------------------------
# 2. Prompts
# ---------------------------------------------------------------------------

def montar_caso(probabilidade: float, limiar: float, fatores: list[dict]) -> dict:
    """Estrutura única que vai para o prompt.

    ``fatores``: lista de dicionários com ``variavel``, ``descricao``, ``valor``
    e ``shap`` (positivo aumenta a chance de SOP). Nenhum identificador de
    paciente entra aqui.
    """
    def _nativo(v):
        v = float(v)
        return int(v) if v.is_integer() else round(v, 3)

    return {
        "probabilidade": round(float(probabilidade), 3),
        "decisao": "SOP provável" if probabilidade >= limiar else "SOP pouco provável",
        "fatores": [{"variavel": f["variavel"], "descricao": f.get("descricao", ""),
                     "valor": _nativo(f["valor"]), "efeito": "aumenta" if f["shap"] > 0 else "reduz",
                     "shap": round(float(f["shap"]), 3)} for f in fatores],
    }


def _bloco_dados(caso: dict, contexto_textual: str | None = None) -> str:
    linhas = [f"Probabilidade estimada pelo modelo: {caso['probabilidade'] * 100:.0f}%",
              f"Decisão do modelo: {caso['decisao']}",
              "Fatores que mais pesaram nesta previsão (SHAP):"]
    for f in caso["fatores"]:
        desc = f" ({f['descricao']})" if f["descricao"] else ""
        linhas.append(f'- "{f["variavel"]}"{desc}: valor {f["valor"]}, {f["efeito"]} a chance '
                      f"(contribuição {f['shap']:+.3f})")
    if contexto_textual:  # base para o Módulo 3 (anotações clínicas, laudos)
        linhas.append(f"Contexto clínico adicional: {contexto_textual}")
    return "\n".join(linhas)


SYSTEM_V2 = """Você é um assistente de apoio à decisão clínica que explica, para médicos e médicas, a saída de um modelo de machine learning de triagem para Síndrome dos Ovários Policísticos (SOP).

Regras obrigatórias:
1. Você NÃO faz diagnóstico. Use linguagem de probabilidade ("o modelo estima", "sugere"), nunca "a paciente tem SOP".
2. Use SOMENTE os dados fornecidos. Não cite exames, sintomas ou valores que não estão na entrada.
3. Cite os fatores pelo nome exato, entre aspas, e diga se cada um aumenta ou reduz a chance.
4. Repita a probabilidade exatamente como foi fornecida, em porcentagem.
5. Termine com a ressalva de que a análise não substitui a avaliação médica.
6. Responda em português, em no máximo 150 palavras."""

EXEMPLOS_V3 = [
    {
        "entrada": {"probabilidade": 0.91, "decisao": "SOP provável", "fatores": [
            {"variavel": "Follicle No. (R)", "descricao": "nº de folículos, ovário direito", "valor": 15, "efeito": "aumenta", "shap": 1.42},
            {"variavel": "Skin darkening (Y/N)", "descricao": "escurecimento da pele", "valor": 1, "efeito": "aumenta", "shap": 0.61},
            {"variavel": "Cycle(R/I)", "descricao": "ciclo regular (2) ou irregular (4)", "valor": 4, "efeito": "aumenta", "shap": 0.55},
            {"variavel": "Weight gain(Y/N)", "descricao": "ganho de peso", "valor": 1, "efeito": "aumenta", "shap": 0.40},
            {"variavel": "Age (yrs)", "descricao": "idade", "valor": 34, "efeito": "reduz", "shap": -0.22}]},
        "saida": {
            "resumo": "O modelo estima 91% de probabilidade de SOP, com decisão de SOP provável.",
            "fatores_principais": ["Follicle No. (R)", "Skin darkening (Y/N)", "Cycle(R/I)", "Weight gain(Y/N)", "Age (yrs)"],
            "confianca": "alta",
            "sugestao_investigacao": "Os sinais mais fortes são ultrassonográficos e clínicos; vale confirmar com a avaliação hormonal e os critérios diagnósticos usados pela equipe.",
            "limitacoes": "Estimativa de um modelo treinado com 541 pacientes de uma única origem. Não substitui a avaliação médica."}},
    {
        "entrada": {"probabilidade": 0.12, "decisao": "SOP pouco provável", "fatores": [
            {"variavel": "Follicle No. (L)", "descricao": "nº de folículos, ovário esquerdo", "valor": 3, "efeito": "reduz", "shap": -0.95},
            {"variavel": "Cycle(R/I)", "descricao": "ciclo regular (2) ou irregular (4)", "valor": 2, "efeito": "reduz", "shap": -0.48},
            {"variavel": "hair growth(Y/N)", "descricao": "crescimento de pelos", "valor": 0, "efeito": "reduz", "shap": -0.30},
            {"variavel": "AMH(ng/mL)", "descricao": "hormônio antimülleriano", "valor": 6.1, "efeito": "aumenta", "shap": 0.18},
            {"variavel": "BMI", "descricao": "índice de massa corporal", "valor": 22.4, "efeito": "reduz", "shap": -0.11}]},
        "saida": {
            "resumo": "O modelo estima 12% de probabilidade de SOP, com decisão de SOP pouco provável.",
            "fatores_principais": ["Follicle No. (L)", "Cycle(R/I)", "hair growth(Y/N)", "AMH(ng/mL)", "BMI"],
            "confianca": "alta",
            "sugestao_investigacao": "Quase todos os fatores reduzem a chance; o AMH vai no sentido contrário e pode merecer atenção se houver queixa clínica.",
            "limitacoes": "Estimativa de um modelo treinado com 541 pacientes de uma única origem. Não substitui a avaliação médica."}},
]


def _instrucao_json() -> str:
    return ("Responda APENAS com um objeto JSON com exatamente estes campos: "
            '"resumo" (texto), "fatores_principais" (lista com os nomes exatos das variáveis), '
            '"confianca" ("alta", "media" ou "baixa"), "sugestao_investigacao" (texto) e '
            '"limitacoes" (texto, incluindo que não substitui a avaliação médica).')


def montar_prompt(versao: str, caso: dict, contexto_textual: str | None = None) -> tuple[str, str]:
    """Devolve ``(system, user)`` para a versão pedida."""
    dados = _bloco_dados(caso, contexto_textual)
    if versao == "v1":
        return "", f"Explique este resultado.\n\n{dados}"
    if versao == "v2":
        return SYSTEM_V2, f"Explique este resultado para o médico.\n\n{dados}"
    if versao == "v3":
        exemplos = []
        for i, ex in enumerate(EXEMPLOS_V3, 1):
            exemplos.append(f"Exemplo {i}\nEntrada:\n{_bloco_dados(ex['entrada'])}\n"
                            f"Saída:\n{json.dumps(ex['saida'], ensure_ascii=False)}")
        system = SYSTEM_V2.replace("6. Responda em português, em no máximo 150 palavras.",
                                   "6. Responda em português.\n7. " + _instrucao_json())
        user = "\n\n".join(exemplos) + f"\n\nAgora o caso real.\nEntrada:\n{dados}\nSaída:"
        return system, user
    raise ValueError(f"versão desconhecida: {versao}")


# ---------------------------------------------------------------------------
# 3. Checagens automáticas
# ---------------------------------------------------------------------------

def normalizar(texto: str) -> str:
    """Minúsculas e sem acento (aula 3 de PLN)."""
    t = unicodedata.normalize("NFKD", (texto or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def extrair_json(texto: str | None):
    """Devolve ``(dados, problema)``. ``problema`` é None quando está tudo certo."""
    t = (texto or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        dados = json.loads(t)
    except json.JSONDecodeError as e:
        return None, f"JSON inválido: {e.msg}"
    if not isinstance(dados, dict):
        return None, "o JSON não é um objeto"
    faltando = [c for c in CAMPOS_V3 if c not in dados]
    tipo_errado = [c for c, tp in CAMPOS_V3.items() if c in dados and not isinstance(dados[c], tp)]
    if faltando or tipo_errado:
        return dados, f"faltando={faltando} tipo_errado={tipo_errado}"
    return dados, None


PADROES_DIAGNOSTICO = [
    r"\b(tem|possui|apresenta|esta com|e portadora de) (a )?(sop|sindrome dos ovarios policisticos)\b",
    r"\bdiagnostico (de sop )?(confirmado|definitivo)\b",
    r"\bconfirma(-se)? (o )?diagnostico\b",
    r"\b(nao tem|nao possui) (a )?(sop|sindrome)\b",
]
PADROES_RESSALVA = [r"nao substitui", r"avaliacao medica", r"julgamento clinico",
                    r"criterio medico", r"avaliacao clinica", r"decisao medica"]


def _texto_para_checar(texto, versao):
    """Na v3 as checagens de texto olham os campos do JSON; nas outras, o texto inteiro."""
    if versao == "v3":
        dados, _ = extrair_json(texto)
        if isinstance(dados, dict):
            return " ".join(str(v) if not isinstance(v, list) else " ".join(map(str, v))
                            for v in dados.values()), dados
    return texto or "", None


def checar_resposta(texto: str | None, caso: dict, versao: str, nomes_conhecidos=()) -> dict:
    """Checagens da seção 4.6 do overview. Cada chave é um booleano (ou número).

    - ``fatores_inventados``: variáveis citadas que não estavam entre as 5 fornecidas.
      Na v3 compara a lista ``fatores_principais``; nas outras procura no texto os
      nomes de todas as colunas do dataset (``nomes_conhecidos``).
    - ``cobertura_fatores``: fração dos 5 fatores fornecidos que aparecem na resposta.
    - ``probabilidade_correta``: toda porcentagem citada bate com a do modelo (±1 ponto).
    - ``diagnostico_definitivo``: afirma que a paciente tem (ou não tem) SOP.
    - ``tem_ressalva``: diz que não substitui a avaliação médica.
    - ``json_valido``: só na v3.
    """
    if not texto:
        return {"respondeu": False}
    alvo, dados = _texto_para_checar(texto, versao)
    alvo_n = normalizar(alvo)
    fornecidos = [f["variavel"] for f in caso["fatores"]]
    fornecidos_n = {normalizar(f) for f in fornecidos}

    if versao == "v3" and dados is not None and isinstance(dados.get("fatores_principais"), list):
        citados = [str(f) for f in dados["fatores_principais"]]
    else:
        citados = [n for n in nomes_conhecidos if normalizar(n) in alvo_n]
    inventados = [c for c in citados if normalizar(c) not in fornecidos_n]

    cobertos = [f for f in fornecidos if normalizar(f) in alvo_n]
    porcentagens = [float(p.replace(",", ".")) for p in re.findall(r"(\d{1,3}(?:[.,]\d+)?)\s*%", alvo)]
    prob = caso["probabilidade"] * 100

    resultado = {
        "respondeu": True,
        "fatores_inventados": inventados,
        "sem_fator_inventado": not inventados,
        "cobertura_fatores": len(cobertos) / len(fornecidos) if fornecidos else 0.0,
        "cita_probabilidade": bool(porcentagens),
        "probabilidade_correta": bool(porcentagens) and all(abs(p - prob) <= 1.0 for p in porcentagens),
        "diagnostico_definitivo": any(re.search(p, alvo_n) for p in PADROES_DIAGNOSTICO),
        "tem_ressalva": any(re.search(p, alvo_n) for p in PADROES_RESSALVA),
    }
    if versao == "v3":
        resultado["json_valido"] = extrair_json(texto)[1] is None
    return resultado


# ---------------------------------------------------------------------------
# 4. Registro das interações (base para o Módulo 3)
# ---------------------------------------------------------------------------

def ler_interacoes(arquivo: Path) -> list[dict]:
    if not Path(arquivo).exists():
        return []
    with open(arquivo, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]


def registrar_interacao(arquivo: Path, registro: dict) -> None:
    Path(arquivo).parent.mkdir(parents=True, exist_ok=True)
    with open(arquivo, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
