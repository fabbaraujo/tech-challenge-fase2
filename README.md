# Tech Challenge Fase 2: otimização dos modelos de SOP com algoritmo genético e interpretação com LLM

Pós-Graduação IA para Devs (FIAP) · **Projeto 1, otimização de modelos de diagnóstico** · Autor: Fabricio Geraldo Araujo

Continuação do modelo de triagem de **Síndrome dos Ovários Policísticos (SOP)** da Fase 1. Os hiperparâmetros da Regressão Logística, do Random Forest e do KNN são otimizados com um **algoritmo genético** próprio, e as previsões do modelo final são explicadas para o médico por uma **LLM pré-treinada**, com três versões de prompt avaliadas automaticamente.

- Repositório da Fase 1 (Grupo 21): [fiap-tech-chalange-1-ia-para-devs](https://github.com/michel005/fiap-tech-chalange-1-ia-para-devs)
- Dataset: [Polycystic ovary syndrome (PCOS), Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos/data), arquivo `PCOS_data_without_infertility.xlsx`, aba `Full_new`
- Vídeo de apresentação: *(link do YouTube)*
- Relatório técnico: [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) (PDF em `docs/relatorio_tecnico.pdf`)

## Resultados

F2 é a fitness do AG: a família do F1 com o recall pesando 4 vezes mais que a precisão (ver relatório, seção 2.3).

| Modelo | F2 em CV, original | F2 em CV, otimizado | Recall no teste, original → otimizado | Falsos negativos no teste |
|---|---|---|---|---|
| Regressão Logística | 0,790 | **0,880** | 0,861 → 0,917 | 5 → 3 |
| Random Forest | 0,787 | **0,852** | 0,778 → 0,833 | 8 → 6 |
| KNN | 0,725 | **0,728** | 0,722 → 0,750 | 10 → 9 |

CV = média da validação cruzada estratificada de 5 dobras no treino (432 pacientes). Teste = 109 pacientes, 36 com SOP, usado uma única vez.

## Estrutura

```
tech-challenge-fase2/
├── notebook/tech_challenge_fase2.ipynb   o projeto inteiro, com a narrativa
├── src/ga.py                             algoritmo genético (codificação, operadores, loop)
├── src/llm.py                            cliente LLM com troca de provedor, prompts e checagens
├── tests/test_ga.py, tests/test_llm.py   testes automatizados (pytest)
├── docs/arquitetura.md                   diagrama e decisões técnicas
├── docs/relatorio_tecnico.md             relatório
├── datasets/                             PCOS_data_without_infertility.xlsx (Kaggle)
└── outputs/                              tabelas, gráficos, runs do AG e interações com o LLM
```

## Como rodar

Requer Python 3.11 ou mais novo.

```bash
python3 -m venv .venv
source .venv/bin/activate          # no fish: source .venv/bin/activate.fish
pip install -r requirements.txt
cp .env.example .env               # e preencha GROQ_API_KEY (console.groq.com, plano gratuito)
```

**Testes:**

```bash
pytest
```

**Notebook:** abra `notebook/tech_challenge_fase2.ipynb` no VS Code ou no Jupyter, selecione o kernel do `.venv` e rode tudo.

- Os 27 runs do AG estão salvos em `outputs/experimentos/` e são carregados. Para refazer, apague a pasta e use `RODAR_EXPERIMENTOS = True` na primeira célula (cerca de 2 horas em 2 núcleos).
- Sem `GROQ_API_KEY`, o notebook roda em modo offline e pula só as chamadas ao LLM.

**MLflow** (depois de rodar os experimentos):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

e abra http://127.0.0.1:5000.

## Provedor de LLM

Padrão: `openai/gpt-oss-120b` no Groq. Para trocar, mude `LLM_PROVIDER` (`groq`, `openai` ou `gemini`) e `LLM_MODEL` no `.env`; o código não muda. OpenAI e Gemini precisam dos pacotes `openai` e `google-genai`.

## Aviso

Ferramenta acadêmica de apoio à decisão. Não faz diagnóstico e não substitui a avaliação médica.
