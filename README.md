# Painel Legislativo: acompanhando proposições e matérias de interesse público

Pipeline de coleta, classificação e visualização de proposições legislativas federais brasileiras
(Câmara dos Deputados, Senado Federal e Congresso Nacional), com foco em temas de interesse público definidos pelo usuário.

O dashboard incluído neste repositório é um **exemplo de aplicação**, construído para o tema de pobreza e desigualdade social (1988–2025).
Os termos de busca, tipos de proposição e demais parâmetros são inteiramente configuráveis via `configs.yaml`.

---

## Estrutura do projeto

```
.
├── main.py               # Pipeline principal (coleta → status → LLM → escala BLS)
├── dashboard.py          # Dashboard interativo (Dash + Plotly)
├── configs.yaml          # Parâmetros editáveis (termos, tipos, modelo LLM, etc.)
├── .env                  # Chave de API Mistral (não versionado)
├── requirements.txt
└── data/                 # Pasta com as fontes de dados (não versionada)
    ├── filtered/         # Parquets anuais das Casas (gerados pelo pipeline)
    ├── output/           # Parquets consolidados por etapa (gerados pelo pipeline)
    └── auxs/
        └── BLS9_full.csv # Brazilian Legislative Surveys (baixar separadamente)
```

---

## Pipeline

O `main.py` executa quatro etapas em sequência, com checkpoint em cada uma:

| Etapa | Arquivo gerado | Descrição |
|---|---|---|
| 1 — Coleta | `etapa1_output_*.parquet` | Coleta via APIs da Câmara e do Senado |
| 2 — Status | `etapa2_enriquecido_*.parquet` | Situação legislativa de cada proposição |
| 3 — LLM | `etapa3_avaliado_*.parquet` | Avaliação de relevância via Mistral AI |
| 4 — BLS | `etapa4_final_*.parquet` | Posicionamento ideológico dos partidos via BLS |

Cada etapa só é executada se o arquivo da etapa anterior existir e o arquivo da etapa atual ainda não tiver sido gerado. Interrupções podem ser retomadas sem reprocessamento.

---

## Configuração

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```
MISTRAL_API_KEY=sua_chave_aqui
```

A chave pode ser obtida em [console.mistral.ai](https://console.mistral.ai).

### 3. Dados do BLS (Brazilian Legislative Surveys)

O enriquecimento ideológico utiliza os dados do **Brazilian Legislative Surveys Waves 1–9 (1990–2021)**, disponíveis no Harvard Dataverse:

> Power, Timothy J.; Zucco, Cesar, 2021, *Brazilian Legislative Surveys (Waves 1-9, 1990-2021)*,
> [doi:10.7910/DVN/WM9IZ8](https://doi.org/10.7910/DVN/WM9IZ8), Harvard Dataverse.

**Termos de uso:** dados disponibilizados sob licença Creative Commons CC0 1.0 com termos adicionais. 
O anonimato dos respondentes é garantido por IRBs da Oxford University, Rutgers University e Fundação Getulio Vargas.
Ao utilizar estes dados, o usuário compromete-se a não tentar identificar os respondentes nominalmente e a citar o dataset e/ou as publicações relacionadas.

**Para baixar:** acesse o link acima, preencha o guestbook e baixe o arquivo `BLS9_full.csv`. Salve-o em `data/auxs/BLS9_full.csv`. 
Para saber mais sobre os atributos da base, é também possível baixar o dicionário `BLS9_Codebook.txt`. 

### 4. Configurar os parâmetros

Edite o `configs.yaml` conforme seu tema de interesse. Os principais parâmetros são:

```yaml
ano_inicio: 1988   # ano de início da varredura de proposições e matérias
ano_fim: 2025      # ano de término da varredura de proposições e matérias

termos:            # adicione os termos relevantes para o seu tema
  - pobreza
  - desigualdade social
  - transferência de renda
  - ...

avaliacao:
  model: ministral-8b-2410   # modelo Mistral a utilizar
  nota_corte: 6.0            # proposições abaixo desta nota são excluídas
  categorias:                # categorias temáticas para classificação LLM
    - Assistência Social
    - Pobreza
    - ...
```

### 5. Executar o pipeline

```bash
python main.py
```

### 6. Executar o dashboard

```bash
python dashboard.py
```

O dashboard estará disponível em `http://localhost:8050`.

---

## APIs utilizadas

- **Câmara dos Deputados:** [dadosabertos.camara.leg.br](https://dadosabertos.camara.leg.br)
- **Senado Federal:** [legis.senado.leg.br/dadosabertos](https://legis.senado.leg.br/dadosabertos)
- **Mistral AI:** [docs.mistral.ai](https://docs.mistral.ai)

---

## Observações metodológicas

- Os **scores ideológicos** são derivados das percepções dos próprios legisladores sobre o posicionamento dos partidos (escala 1–10, onde 1 = esquerda e 10 = direita), conforme metodologia do BLS.
- Para partidos sem cobertura direta no BLS, são utilizados **proxies ideológicos** documentados no código (`_PROXIES_BLS` em `main.py`).
- A **avaliação de relevância** via LLM é configurável: o prompt, os critérios de inclusão/exclusão e a escala de notas podem ser ajustados em `configs.yaml`.
- O dashboard incluído é um **exemplo** construído para o tema de pobreza e desigualdade. Qualquer tema de interesse público pode ser configurado seguindo a mesma estrutura.

---
 
## Inteligência Artificial
 
Este projeto foi desenvolvido com o auxílio de Inteligência Artificial Generativa (GenAI), utilizado como assistente de programação ao longo de todo o desenvolvimento do código Python — arquitetura do pipeline, decisões metodológicas, debugging e construção do dashboard. A ferramenta utilizada foi:

> ANTHROPIC. Claude Sonnet 4.6. San Francisco: Anthropic, 2025. Disponível em: https://claude.ai. Acesso em: 27 abr. 2026.

O uso de IA não substitui a responsabilidade intelectual do autor sobre as escolhas metodológicas, interpretações e resultados apresentados. Todo o código foi revisado, validado e é de responsabilidade do autor. 

A etapa de avaliação de relevância das proposições utiliza o modelo **Mistral `ministral-8b-2410`** (Mistral AI), responsável pela leitura das ementas, classificação temática e atribuição de notas de relevância conforme rubrica definida pelo autor em `configs.yaml`.

> [!WARNING]
> #### Aviso de segurança — ataque Shai-Hulud (maio/2026)
> Em 12 de maio de 2026, a versão `mistralai==2.4.6` foi comprometida por um ataque de supply chain conhecido como **Mini Shai-Hulud** (grupo TeamPCP).
> A versão maliciosa continha um credential stealer que executava automaticamente ao importar o pacote em sistemas Linux.
> Até a presente data, a última versão segura é **`mistralai==2.4.5`**, que é a versão pinada neste projeto.
>
> Verifique se você não está afetado:
> `grep -n 'mistralai.*2\.4\.6' requirements*.txt pyproject.toml uv.lock poetry.lock 2>/dev/null`
>
> Mais informações: [advisory oficial da Mistral](https://docs.mistral.ai/resources/security-advisories) 
---

## Autor

**Pier Francesco De Maria**

Cientista de dados, Doutor em Demografia e pesquisador (pobreza, desigualdade e políticas públicas)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-dpierf-blue?logo=linkedin)](https://www.linkedin.com/in/dpierf)
[![Email](https://img.shields.io/badge/Email-dpierf%40gmail.com-red?logo=gmail)](mailto:dpierf@gmail.com)
