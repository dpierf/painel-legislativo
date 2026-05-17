# == 0. Importação de Pacotes ===================================================

import re
import os
import json
import yaml
import requests
import numpy            as np
import pandas           as pd
from tqdm               import tqdm
from time               import sleep
from dotenv             import load_dotenv
from pathlib            import Path
from threading          import Lock
from mistralai.client   import Mistral
from collections        import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()


# == 1. Configuração ============================================================

# ── Parametrização ─────────────────────────────────────────────────────────────
_cfg = yaml.safe_load(
    Path('configs.yaml').read_text(encoding='utf-8')
)

TERMOS                  = _cfg['termos']
TIPOS_CAMARA            = _cfg['tipos_camara']
TIPOS_SENADO            = set(_cfg['tipos_senado'])
TIPOS_CONTEUDO_SENADO   = set(_cfg['tipos_conteudo_senado'])
ASSUNTOS_EXCLUIR        = set(_cfg['assuntos_excluir'])
ANO_INICIO              = _cfg['ano_inicio']
ANO_FIM                 = _cfg['ano_fim']
SLEEP_CAMARA            = _cfg['sleep_camara']
SLEEP_SENADO            = _cfg['sleep_senado']
MAX_WORKERS_COMBOS      = _cfg['max_workers_combos']
MAX_WORKERS_AUTORIA     = _cfg['max_workers_autoria']

TIPO_NORM = {
    # Câmara (siglas)
    'PL':  'PL',   'PLP': 'PLP',  'PEC': 'PEC',  'MPV': 'MPV',
    'PDL': 'PDL',  'PDC': 'PDL',  # PDC = código antigo de PDL
    'PLV': 'PLV',  'PLS': 'PLS',  'PLC': 'PLC',  'PLN': 'PLN',
    'PRC': 'PRC',  'PRN': 'PRN',
    'MSC': 'MSG',  'MSG': 'MSG',
    'PFC': 'PFC',  'SUG': 'SUG',
    # Senado (descrições por extenso)
    'Medida Provisória':                   'MPV',
    'Projeto de Lei Ordinária':            'PL',
    'Projeto de Lei da Câmara':            'PLC',
    'Projeto de Lei Complementar':         'PLP',
    'Projeto de Lei do Senado':            'PLS',
    'Projeto de Lei de Conversão':         'PLV',
    'Projeto de Decreto Legislativo':      'PDL',
    'Projeto de Resolução':                'PRC',
    'Proposta de Emenda à Constituição':   'PEC',
    'Proposta de Emenda Constitucional':   'PEC',
    'Mensagem':                            'MSG',
    'Proposta de Fiscalização e Controle': 'PFC',
    'Sugestão':                            'SUG',
}

EXCLUIR_EMENTA = re.compile(
    _cfg['excluir_ementa_regex'],
    re.IGNORECASE
)

# ── Normalização de autoria ────────────────────────────────────────────────────
TIPO_MAP = {
    'Deputado':             None,
    'Deputada':             None,
    'Deputado(a)':          None,
    'Senador':              None,
    'Senadora':             None,
    'Senador(a)':           None,
    'Órgão':                'Comissão',
    'Executivo':            'Executivo',
    'Senado Federal':       'Legislativo',
    'Poder Judiciário':     'Judiciário',
    'Tribunal de Contas':   'TCU',
    'Ministério Público':   'MPU',
    'Sociedade Civil':      'Iniciativa Popular',
}

def _normaliza_autor(nome, partido, tipo_bruto):
    tipo_norm = TIPO_MAP.get(tipo_bruto, tipo_bruto)
    if partido:
        return nome, partido
    elif tipo_norm is None:
        return nome, 'Sem partido'
    else:
        return nome, tipo_norm


# == 2. Câmara dos Deputados ====================================================

def _autores_camara(id_prop, tipo_prop):
    url = f'https://dadosabertos.camara.leg.br/api/v2/proposicoes/{id_prop}/autores'
    try:
        dados = requests.get(url, timeout=10).json().get('dados', [])
        if not dados:
            return {'parlamentar': '', 'uf': '', 'tipo_autor': ''}

        parl_list, tipo_list, uf_list = [], [], []
        for a in dados:
            nome    = a.get('nome', '')
            partido = a.get('siglaPartido', '')
            uf      = a.get('siglaUf', '')
            tipo_br = a.get('tipo', '')

            # fallback: busca partido e UF no perfil do parlamentar
            # acionado quando a API omite esses campos (ocorre em PDLs e outros)
            if not partido and a.get('uri') and tipo_br in (
                'Deputado(a)', 'Deputado', 'Senador(a)', 'Senador'
            ):
                try:
                    perfil  = requests.get(a['uri'], timeout=8).json()
                    perfil_dados = perfil.get('dados', {}).get('ultimoStatus', {})
                    partido = perfil_dados.get('siglaPartido', '')
                    uf      = perfil_dados.get('siglaUf', uf)
                except Exception:
                    pass

            parl_norm, tipo_norm = _normaliza_autor(nome, partido, tipo_br)
            parl_list.append(parl_norm)
            tipo_list.append(tipo_norm)
            if uf:
                uf_list.append(uf)

        tipo_autor = ('Popular; Comissão'
                      if tipo_prop == 'SUG'
                      else '; '.join(dict.fromkeys(tipo_list)))

        return {
            'parlamentar': '; '.join(parl_list),
            'uf':          '; '.join(uf_list),
            'tipo_autor':  tipo_autor,
        }
    except Exception:
        return {'parlamentar': '', 'uf': '', 'tipo_autor': ''}


def _fetch_combo_camara(termo, tipo, ano):
    base      = 'https://dadosabertos.camara.leg.br/api/v2/proposicoes'
    resultado = []
    pagina    = 1
    while True:
        try:
            dados = requests.get(base, params={
                'siglaTipo': tipo, 'ano': ano, 'keywords': termo,
                'itens': 100, 'pagina': pagina,
                'ordem': 'ASC', 'ordenarPor': 'id',
            }, timeout=15).json().get('dados', [])
        except Exception as e:
            tqdm.write(f'  [ERRO CD] {tipo} {ano} "{termo}": {e}')
            break
        if not dados:
            break
        for prop in dados:
            resultado.append({
                'origem':             'CD',
                'tipo':               prop.get('siglaTipo', tipo),
                'numero':             prop.get('numero', ''),
                'ano':                str(prop.get('ano', ano)),
                'data_apre':          prop.get('dataApresentacao', ''),
                'ementa':             prop.get('ementa', ''),
                'id_interno':         prop['id'],
                'id_processo':        '',
                'uri':                prop.get('uri', ''),
                'termo_busca':        termo,
                'tramitando':         '',
                'assunto_geral':      '',
                'assunto_especifico': '',
            })
        pagina += 1
        sleep(SLEEP_CAMARA)
    return resultado


def coletar_camara(termos, tipos, ano_inicio, ano_fim):
    combos    = [(t, tp, a) for t in termos for tp in tipos
                            for a in range(ano_inicio, ano_fim + 1)]
    registros = {}
    lock      = Lock()

    # coleta paralela de combos
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_COMBOS) as ex:
        futuros = {ex.submit(_fetch_combo_camara, t, tp, a): (t, tp, a)
                   for t, tp, a in combos}
        for fut in tqdm(as_completed(futuros), total=len(futuros),
                        desc='CD combos', unit='combo'):
            for prop in fut.result():
                pid = prop['id_interno']
                with lock:
                    if pid not in registros:
                        registros[pid] = prop

    df = pd.DataFrame(registros.values())
    if df.empty:
        return df

    # enriquecimento paralelo de autorias
    pid_tipo     = df.set_index('id_interno')['tipo'].to_dict()
    autoria      = {}
    autoria_lock = Lock()

    def _enrich(pid):
        r = _autores_camara(pid, pid_tipo[pid])
        sleep(SLEEP_CAMARA)
        with autoria_lock:
            autoria[pid] = r

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_AUTORIA) as ex:
        futuros = {ex.submit(_enrich, pid): pid for pid in df['id_interno']}
        for fut in tqdm(as_completed(futuros), total=len(futuros),
                        desc='CD autoria', unit='prop'):
            fut.result()

    df_autoria = pd.DataFrame([autoria[pid] for pid in df['id_interno']])
    return pd.concat([df.reset_index(drop=True), df_autoria], axis=1)


# == 3. Senado Federal / Congresso Nacional =====================================

def _busca_detalhe_senado(id_processo):
    url = f'https://legis.senado.leg.br/dadosabertos/processo/{id_processo}'
    try:
        d    = requests.get(url, headers={'Accept': 'application/json'},
                            timeout=10).json()
        cont = d.get('conteudo', {})
        doc  = d.get('documento', {})
        return (
            doc.get('indexacao',           '') or '',
            cont.get('assuntoGeral',       '') or '',
            cont.get('assuntoEspecifico',  '') or '',
        )
    except Exception:
        return '', '', ''


def _parse_autor_senado(autoria_str, tipo_doc):
    if tipo_doc == 'Sugestão':
        return autoria_str, 'Popular; Comissão', ''

    match = re.match(r'^Senador[a]?\s+(.+?)\s+\((\w+)/(\w+)\)$',
                     autoria_str.strip())
    if match:
        nome, partido, uf = match.groups()
        parl_norm, tipo_norm = _normaliza_autor(nome, partido, 'Senador')
        return parl_norm, tipo_norm, uf

    s = autoria_str.lower()
    if any(t in s for t in ['presidência', 'executivo', 'governo federal']):
        tipo_br = 'Executivo'
    elif 'comissão' in s or 'comissao' in s:
        tipo_br = 'Órgão'
    elif any(t in autoria_str for t in ['STF', 'STJ', 'TSE', 'TST']):
        tipo_br = 'Poder Judiciário'
    elif 'ministério público' in s:
        tipo_br = 'Ministério Público'
    elif 'câmara' in s or 'senado' in s:
        tipo_br = 'Senado Federal'
    else:
        tipo_br = 'Órgão'

    parl_norm, tipo_norm = _normaliza_autor(autoria_str, '', tipo_br)
    return parl_norm, tipo_norm, ''


def _parse_processo_senado(p, termo_busca=''):
    tipo_doc                    = p.get('tipoDocumento', '')
    parlamentar, tipo_autor, uf = _parse_autor_senado(
        p.get('autoria', ''), tipo_doc
    )
    origem = p.get('casaIdentificadora', 'SF')

    return {
        'origem':             origem,
        'tipo':               tipo_doc,
        'numero':             p.get('identificacao', ''),
        'ano':                (p.get('dataApresentacao', '') or '')[:4],
        'data_apre':          p.get('dataApresentacao', ''),
        'ementa':             (p.get('ementa', '') or '').strip(),
        'id_interno':         f'SF-{p.get("codigoMateria", "")}',
        'id_processo':        str(p.get('id', '')),
        'uri':                f'http://legis.senado.leg.br/dadosabertos/materia/{p.get("codigoMateria","")}',
        'termo_busca':        termo_busca,
        'tramitando':         p.get('tramitando', ''),
        'parlamentar':        parlamentar,
        'uf':                 uf,
        'tipo_autor':         tipo_autor,
        'assunto_geral':      '',
        'assunto_especifico': '',
    }


def coletar_senado(termos, tipos_senado, ano_inicio, ano_fim):
    base    = 'https://legis.senado.leg.br/dadosabertos/processo'
    headers = {'Accept': 'application/json'}
    padrao  = re.compile('|'.join(re.escape(t) for t in termos), re.IGNORECASE)
    registros    = {}
    lock         = Lock()
    para_detalhe = []   
    por_ementa   = {}   

    # ── Passo 1: lista de candidatos por ano + filtro rápido por ementa ───────
    for ano in tqdm(range(ano_inicio, ano_fim + 1), desc='SF/CN lista', unit='ano'):
        try:
            todos = requests.get(base, params={'ano': ano},
                                 headers=headers, timeout=30).json()
        except Exception as e:
            tqdm.write(f'  [ERRO SF/CN] {ano}: {e}')
            sleep(SLEEP_SENADO)
            continue

        for p in todos:
            if p.get('tipoDocumento', '') not in tipos_senado:
                continue
            cod = str(p.get('codigoMateria', ''))
            if not cod:
                continue

            ementa = p.get('ementa', '') or ''
            match  = padrao.search(ementa)
            if match:
                if cod not in registros:
                    registros[cod] = _parse_processo_senado(
                        p, match.group(0).lower()
                    )
                    por_ementa[cod] = p.get('id', '')
            else:
                if p.get('tipoConteudo', '') in TIPOS_CONTEUDO_SENADO:
                    para_detalhe.append(p)

        sleep(SLEEP_SENADO)

    print(f'  Capturados por ementa:   {len(registros)}')
    print(f'  Candidatos p/ detalhe:   {len(para_detalhe)}')

    # ── Passo 2: indexacao para não capturados (paralelo) ─────────────────────
    def _enrich_indexacao(p):
        cod     = str(p.get('codigoMateria', ''))
        id_proc = p.get('id', '')
        indexacao, assunto_g, assunto_e = _busca_detalhe_senado(id_proc)
        match = padrao.search(indexacao)
        if match and cod:
            rec = _parse_processo_senado(p, match.group(0).lower())
            rec['assunto_geral']      = assunto_g
            rec['assunto_especifico'] = assunto_e
            with lock:
                if cod not in registros:
                    registros[cod] = rec
        sleep(0.1)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_AUTORIA) as ex:
        futuros = {ex.submit(_enrich_indexacao, p): p for p in para_detalhe}
        for fut in tqdm(as_completed(futuros), total=len(futuros),
                        desc='SF/CN indexacao', unit='mat'):
            fut.result()

    # ── Passo 3: assunto para capturados por ementa ───────────────────────────
    def _enrich_assunto(cod, id_proc):
        _, assunto_g, assunto_e = _busca_detalhe_senado(id_proc)
        with lock:
            if cod in registros:
                registros[cod]['assunto_geral']      = assunto_g
                registros[cod]['assunto_especifico'] = assunto_e
        sleep(0.1)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_AUTORIA) as ex:
        futuros = {ex.submit(_enrich_assunto, cod, id_proc): cod
                   for cod, id_proc in por_ementa.items()}
        for fut in tqdm(as_completed(futuros), total=len(futuros),
                        desc='SF/CN assunto', unit='mat'):
            fut.result()

    return pd.DataFrame(registros.values())


# == 4. Enriquecimento de Status Legislativo ====================================

def _status_cd(id_interno):
    '''
    Busca status e norma gerada para proposição da Câmara.
    Endpoint: /proposicoes/{id}
    '''
    url = f'https://dadosabertos.camara.leg.br/api/v2/proposicoes/{id_interno}'
    try:
        dados  = requests.get(url, timeout=10).json().get('dados', {})
        status = dados.get('statusProposicao', {})
        return {
            'situacao':       status.get('descricaoSituacao', ''),
            'norma_tipo':     '',
            'norma_numero':   '',
            'norma_ano':      '',
            'norma_data_pub': '',
            'norma_url':      status.get('url', ''),
        }
    except Exception:
        return {k: '' for k in ['situacao', 'norma_tipo', 'norma_numero',
                                 'norma_ano', 'norma_data_pub', 'norma_url']}


def _status_sf(id_interno):
    '''
    Busca situação do Senado via v7 (deprecated mas ainda ativo).
    Usa codigo_materia extraído do ID Interno (SF-XXXXX → XXXXX).
    '''
    codigo = id_interno.replace('SF-', '')
    url    = f'http://legis.senado.leg.br/dadosabertos/materia/situacaoatual/{codigo}'
    try:
        d = requests.get(url, headers={'Accept': 'application/json'},
                         timeout=10).json()

        materias = (d.get('SituacaoAtualMateria', {})
                     .get('Materias', {})
                     .get('Materia', []))
        if isinstance(materias, dict):
            materias = [materias]
        if not materias:
            return {k: '' for k in ['situacao', 'norma_tipo', 'norma_numero',
                                     'norma_ano', 'norma_data_pub', 'norma_url']}

        autuacoes = (materias[0]
                     .get('SituacaoAtual', {})
                     .get('Autuacoes', {})
                     .get('Autuacao', []))
        if isinstance(autuacoes, dict):
            autuacoes = [autuacoes]

        # percorre todas as autuações e pega a última situação
        situacao_desc = ''
        situacao_data = ''
        for aut in autuacoes:
            sits = aut.get('Situacoes', {}).get('Situacao', [])
            if isinstance(sits, dict):
                sits = [sits]
            for sit in sits:
                situacao_desc = sit.get('DescricaoSituacao', situacao_desc)
                situacao_data = sit.get('DataSituacao',      situacao_data)

        return {
            'situacao':       situacao_desc,
            'norma_tipo':     '',
            'norma_numero':   '',
            'norma_ano':      '',
            'norma_data_pub': situacao_data,
            'norma_url':      '',
        }
    except Exception:
        return {k: '' for k in ['situacao', 'norma_tipo', 'norma_numero',
                                  'norma_ano', 'norma_data_pub', 'norma_url']}


def harmoniza_situacao(s):
    s = str(s).upper().strip()
    if not s or s == 'NAN':
        return 'Sem informação'
    if any(x in s for x in ['NORMA', 'PROMULGAD', 'SANCIONAD', 'CONHECIDA',
                              'APROVADO', 'VETO DELIBERADO']):
        return 'Promulgada'
    elif any(x in s for x in ['ARQUIVAD', 'PREJUDICAD', 'RETIRAD',
                               'SEM EFICÁCIA', 'PERDEU A EFICÁCIA',
                               'REJEITADA', 'TRAMITAÇÃO ENCERRADA',
                               'TRAMITAÇÃO FINALIZADA', 'VETADA',
                               'REVOGADA', 'AUTUAÇÃO FINALIZADA',
                               'DESMEMBRADA', 'DEVOLVIDA']):
        return 'Arquivada'
    elif any(x in s for x in ['AGUARDANDO', 'PAUTA', 'CONJUNTO', 'RELATOR',
                               'REMETIDA', 'TRANSFORMADA EM PROJETO',
                               'TRANSFORMADO EM NOVA', 'EM TRAMITAÇÃO',
                               'PRONTO PARA', 'PEDIDO DE VISTA',
                               'APROVADO O SUBSTITUTIVO',
                               'EM TRAMITAÇÃO ANTES']):
        return 'Em tramitação'
    else:
        return 'Outro'


def enriquecer_status(df, sleep_cd=0.2, sleep_sf=0.2):
    '''
    Enriquece o parquet final com status legislativo.
    CD  : /proposicoes/{id}         → situação + URL da norma se houver
    SF/CN: /processo/{id_processo}  → norma gerada se houver
    Retorna DataFrame com 6 colunas adicionais.
    '''
    COLS_STATUS = ['situacao', 'norma_tipo', 'norma_numero',
                   'norma_ano', 'norma_data_pub', 'norma_url']

    resultados = {}
    lock       = Lock()

    # ── Câmara ─────────────────────────────────────────────────────────────────
    df_cd = df[df['Origem'] == 'CD'].copy()

    def _enrich_cd(idx, id_interno):
        res = _status_cd(id_interno)
        sleep(sleep_cd)
        with lock:
            resultados[idx] = res

    print(f'  CD: {len(df_cd)} proposições')
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_AUTORIA) as ex:
        futuros = {
            ex.submit(_enrich_cd, idx, row['ID Interno']): idx
            for idx, row in df_cd.iterrows()
        }
        for fut in tqdm(as_completed(futuros), total=len(futuros),
                        desc='Status CD', unit='prop'):
            fut.result()

    # ── Senado / CN ────────────────────────────────────────────────────────────
    df_sfcn = df[df['Origem'].isin(['SF', 'CN'])].copy()
    # descarta registros sem ID Processo válido
    df_sfcn = df_sfcn[df_sfcn['ID Processo'].notna() &
                      (df_sfcn['ID Processo'] != '—') &
                      (df_sfcn['ID Processo'] != '')]

    def _enrich_sf(idx, id_interno):
        res = _status_sf(id_interno)
        sleep(sleep_sf)
        with lock:
            resultados[idx] = res

    print(f'  SF/CN: {len(df_sfcn)} matérias')
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_AUTORIA) as ex:
        futuros = {
            ex.submit(_enrich_sf, idx, row['ID Interno']): idx
            for idx, row in df_sfcn.iterrows()
        }
        for fut in tqdm(as_completed(futuros), total=len(futuros),
                        desc='Status SF/CN', unit='mat'):
            fut.result()

    # ── monta colunas de status ────────────────────────────────────────────────
    vazio = {k: '' for k in COLS_STATUS}
    df_status = pd.DataFrame(
        [resultados.get(i, vazio) for i in df.index],
        index=df.index
    )

    # ── harmonização e limpeza ─────────────────────────────────────────────────
    df_resultado = pd.concat([df, df_status], axis=1)
    df_resultado['situacao_norm'] = df_resultado['situacao'].apply(harmoniza_situacao)
    df_resultado = df_resultado.drop(
        columns=['norma_tipo', 'norma_numero', 'norma_ano',
                 'norma_data_pub', 'norma_url']
    )
    return df_resultado


# == 5. Avaliação LLM ===========================================================

_AV_CFG    = _cfg['avaliacao']
_AV_SIGLAS = _cfg['avaliacao'].get('siglas', {})
_AV_MODEL  = _AV_CFG['model']
_AV_SLEEP  = _AV_CFG['sleep']
_AV_CORTE  = _AV_CFG['nota_corte']
_AV_CATS   = _AV_CFG['categorias']
_AV_ESCALA = _AV_CFG['escala']
_AV_INC    = _AV_CFG['inclusao']
_AV_EXC    = _AV_CFG['exclusao']

_mistral   = Mistral(api_key=os.environ['MISTRAL_API_KEY'])


def _busca_contexto_cd(uri):
    '''Busca ementa detalhada e keywords da Câmara via URI.'''
    try:
        d = requests.get(uri, timeout=10).json().get('dados', {})
        return {
            'keywords':        d.get('keywords', '')        or '',
            'ementa_detalhada': d.get('ementaDetalhada', '') or '',
        }
    except Exception:
        return {'keywords': '', 'ementa_detalhada': ''}


def _busca_contexto_sf(uri):
    '''Busca indexação do Senado via URI.'''
    try:
        d = requests.get(uri, headers={'Accept': 'application/json'},
                         timeout=10).json()
        mat = (d.get('DetalheMateria', {})
                .get('Materia', {}))
        idx = (mat.get('IndexacaoMateria', '') or
               mat.get('EmentaMateria',    '') or '')
        return {'indexacao': idx}
    except Exception:
        return {'indexacao': ''}


def _monta_prompt(row, contexto):
    cats = '\n'.join(f'  - {c}' for c in _AV_CATS)
    inc  = '\n'.join(f'  - {c}' for c in _AV_INC)
    exc  = '\n'.join(f'  - {c}' for c in _AV_EXC)

    proposta = f'{row["Tipo de Proposta"]} {row["Número"]}/{row["Ano"]}'

    linhas_ctx = [
        f'Casa:      {row["Origem"]}',
        f'Proposta:  {proposta}',
        f'Ementa:    {row["Ementa"]}',
    ]
    if contexto.get('keywords'):
        linhas_ctx.append(f'Indexação: {contexto["keywords"]}')
    if contexto.get('ementa_detalhada'):
        linhas_ctx.append(f'Ementa detalhada: {contexto["ementa_detalhada"]}')
    if contexto.get('indexacao'):
        linhas_ctx.append(f'Indexação: {contexto["indexacao"]}')

    ctx = '\n'.join(linhas_ctx)
    siglas_fmt = '\n'.join(f'  - {sig} ({desc})' for sig, desc in _AV_SIGLAS.items())

    return f'''
Você é um especialista em políticas públicas de combate à pobreza e desigualdade social no Brasil.
Avalie a relevância da proposição legislativa abaixo para o tema de pobreza e desigualdade social, atribuindo uma nota de 0 a 10 (múltiplos de 0.25).

PROPOSIÇÃO:
{ctx}

═══════════════════════════════════════════════════════════
ESCALA DE RELEVÂNCIA
═══════════════════════════════════════════════════════════
Abaixo, você recebeu exemplos concretos para cada grupo de notas. Use os exemplos abaixo para aperfeiçoar seu processo decisório. Lembre-se que estes são apenas EXEMPLOS.

9.0–10.0: Objeto central e exclusivo. O tema é a razão de ser da proposição. Não existe outra interpretação possível. Exemplos:
    • "Institui o Programa Bolsa Família e define critérios de elegibilidade e valores de transferência de renda para famílias em situação de pobreza"
    • "Altera os critérios de acesso ao BPC para ampliar cobertura a idosos e pessoas com deficiência em extrema pobreza"

7.5–8.75: Objeto principal, com elementos secundários. A proposição trata centralmente do tema, mas abrange aspectos adicionais. Exemplos:
    • "Dispõe sobre o Estatuto do Idoso, estabelecendo direitos, proteções sociais e benefícios assistenciais para pessoas com 60 anos ou mais"
    • "Altera a LOAS para incluir portadores de esclerose lateral amiotrófica no BPC, independentemente do critério de renda familiar"

6.0–7.25: Relevante e substantivo, mas não exclusivo. O tema é um componente importante da proposição, ainda que coexista com outros objetos. Exemplos:
    • "Altera a Lei de Responsabilidade Fiscal para excluir do limite de despesas de pessoal os recursos destinados ao Programa Saúde da Família e aos Centros de Referência de Assistência Social (CRAS)"
    • "Institui o Plano Nacional de Juventude, com eixos de educação, trabalho e inclusão social para jovens em situação de vulnerabilidade"

4.5–5.75: Elemento importante, mas não central. O tema de pobreza ou desigualdade é abordado, mas divide espaço com outros objetos de igual ou maior peso. Exemplos:
    • "Institui o Estatuto da Micro e Pequena Empresa, com benefícios tributários que favorecem empreendedores de baixa renda e trabalhadores informais"
    • "Altera o Código de Defesa do Consumidor para definir critérios de essencialidade de produtos para pessoas com deficiência"

2.0–4.0: Menção tangencial ou instrumental. O tema aparece como justificativa, objetivo genérico ou efeito colateral esperado de uma política setorial com outro foco central. Exemplos:
    • "Destina percentual dos royalties de petróleo a municípios para erradicação da miséria e redução de desigualdades regionais" — pobreza é objetivo declarado genérico demais
    • "Dispõe sobre a Política Nacional de Energia Nuclear, tendo entre seus objetivos a soberania, o desenvolvimento nacional e a erradicação do estado de pobreza" — pobreza como argumento retórico

0.0–1.75: Ausente ou meramente retórico. O tema não é objeto da proposição. Aparece apenas na indexação interna, em preâmbulos genéricos ou não aparece. Exemplos:
    • "Concede a Medalha do Mérito Legislativo ao servidor público destacado"
    • "Dispõe sobre a Política Nacional de Ciência e Tecnologia e o fomento à pesquisa de ponta" — pobreza foi capturada por indexação indireta

═══════════════════════════════════════════════════════════
ATENÇÃO — TEMA ACESSÓRIO OU RETÓRICO
═══════════════════════════════════════════════════════════
Atribua nota ABAIXO DE 5.0 quando identificar qualquer um dos seguintes padrões:
- Pobreza ou desigualdade aparecem apenas na justificativa ou nos "considerandos" da proposição, sem se refletir no dispositivo normativo
- A proposição trata de política setorial (energia, infraestrutura, agronegócio, segurança pública) e menciona redução da pobreza como objetivo genérico de desenvolvimento nacional
- O mecanismo proposto é fiscal ou distributivo (royalties, fundos, incentivos tributários) e a destinação a populações pobres é uma entre várias finalidades possíveis
- A proposição foi capturada pelo sistema de busca por conter os termos na indexação interna, mas a ementa e o conteúdo tratam de outro assunto

═══════════════════════════════════════════════════════════
CATEGORIAS TEMÁTICAS
═══════════════════════════════════════════════════════════
{cats}

REGRA OBRIGATÓRIA: NUNCA invente categorias fora desta lista. Escolha EXATAMENTE uma das categorias acima. 
Se a proposição não se encaixar com clareza em nenhuma delas, preencha obrigatoriamente com: "Indefinido - Revisão Humana".

═══════════════════════════════════════════════════════════
CRITÉRIOS
═══════════════════════════════════════════════════════════
Critérios de Inclusão: {inc}
Critérios de Exclusão: {exc}

═══════════════════════════════════════════════════════════
PROGRAMAS E BENEFÍCIOS DE REFERÊNCIA
═══════════════════════════════════════════════════════════
As siglas e programas abaixo são instrumentos centrais de proteção social no Brasil. 
Proposições que os mencionem diretamente no dispositivo normativo recebem nota SUPERIOR A 7.0, salvo motivo extremamente forte (sigla usada claramente como retórica ou pretexto).
{siglas_fmt}

═══════════════════════════════════════════════════════════

Retorne APENAS um JSON válido, sem texto adicional:
{{
    "nota": <número entre 0 e 10, múltiplo de 0.25>,
    "categoria": "<categoria da lista ou 'Indefinido - Revisão Humana'>",
    "justificativa": "<uma frase explicando a nota>"
}}
'''


def _chama_llm(prompt):
    try:
        resp = _mistral.chat.complete(
            model=_AV_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            response_format={'type': 'json_object'},
        )
        raw  = resp.choices[0].message.content
        data = json.loads(raw)

        # garante múltiplo de 0.25
        nota = round(float(data.get('nota', 0)) * 4) / 4
        nota = max(0.0, min(10.0, nota))

        decisao = 'incluir' if nota >= _AV_CORTE else 'excluir'

        return {
            'llm_nota':          nota,
            'llm_categoria':     str(data.get('categoria', '') or ''),
            'llm_decisao':       decisao,   # calculado pelo código, não pelo modelo
            'llm_justificativa': str(data.get('justificativa', '') or ''),
        }
    except Exception as e:
        tqdm.write(f'  [ERRO LLM] {e}')
        return {
            'llm_nota':          None,
            'llm_categoria':     '',
            'llm_decisao':       '',
            'llm_justificativa': '',
        }


def avaliar_llm(df, ano_inicio, ano_fim):
    '''
    Avalia cada proposição via LLM e salva etapa3_avaliado_{ano_inicio}_{ano_fim}.parquet.
    Retoma de onde parou se o arquivo já existir.
    '''
    saida = DIR_OUT / f'etapa3_avaliado_{ano_inicio}_{ano_fim}.parquet'

    # ── retomada ──────────────────────────────────────────────────────────────
    cols_llm = ['llm_nota', 'llm_categoria', 'llm_decisao', 'llm_justificativa']

    if saida.exists():
        df_prev = pd.read_parquet(saida)
        pendentes = df_prev[df_prev['llm_nota'].isna()].index.tolist()
        tqdm.write(f'  Retomando: {len(pendentes)} pendentes de {len(df_prev)}')
    else:
        df_prev = df.copy()
        for col in cols_llm:
            df_prev[col] = None
        pendentes = df_prev.index.tolist()

    print(f'  Total a avaliar: {len(pendentes)}')

    # ── loop principal ────────────────────────────────────────────────────────
    for idx in tqdm(pendentes, desc='Avaliação LLM', unit='prop'):
        row = df_prev.loc[idx]
        uri = str(row.get('URI', ''))

        # busca contexto adicional
        if row['Origem'] == 'CD':
            ctx = _busca_contexto_cd(uri)
        else:
            ctx = _busca_contexto_sf(uri)

        prompt    = _monta_prompt(row, ctx)
        
        # retry com backoff
        for tentativa in range(3):
            resultado = _chama_llm(prompt)
            if resultado['llm_nota'] is not None:
                break
            tqdm.write(f'  [RETRY {tentativa+1}/3] idx={idx}')
            sleep(_AV_SLEEP * (tentativa + 1) * 3)

        for col, val in resultado.items():
            df_prev.loc[idx, col] = val

        sleep(_AV_SLEEP)

        # checkpoint a cada 100 registros
        if (pendentes.index(idx) + 1) % 100 == 0:
            df_prev.to_parquet(saida, index=False)

    # salvamento final
    for col in cols_llm:
        df_prev[col] = df_prev[col].astype(str).replace('None', '')

    df_prev.to_parquet(saida, index=False)

    print(f'\nArquivo: {saida}')
    print(f'Incluídas (nota >= {_AV_CORTE}): '
          f'{(df_prev["llm_decisao"] == "incluir").sum()}')
    print(f'Excluídas: {(df_prev["llm_decisao"] == "excluir").sum()}')
    print(f'\nDistribuição de categorias:')
    print(df_prev['llm_categoria'].value_counts().to_string())


# == 6. Enriquecimento Ideológico (BLS) ========================================

_BLS_PATH = Path('data/auxs/BLS9_full.csv')

_LR_COLS = {
    'lrpt':     'PT',        'lrpsdb':   'PSDB',      'lrpmdb':   'PMDB/MDB',
    'lrmdb':    'PMDB/MDB',  'lrpfl':    'PFL',       'lrdem':    'DEM',
    'lrpl':     'PL',        'lrpp_ppb': 'PP',        'lrpp':     'PP',
    'lrpdt':    'PDT',       'lrpsb':    'PSB',       'lrpcdob':  'PCdoB',
    'lrptb':    'PTB',       'lrpsol':   'PSOL',      'lrpsd':    'PSD',
    'lrprn':    'PRN',       'lrpps':    'PPS',       'lrpr':     'PR',
    'lrpros':   'PROS',      'lrrede':   'REDE',      'lrpode':   'PODEMOS',
    'lrnovo':   'NOVO',      'lrrep':    'REPUBLICANOS', 'lrpsc':  'PSC',
    'lrpv':     'PV',        'lrprb':    'PRB',       'lrpstu':   'PSTU',
    'lrpcb':    'PCB',       'lrpsl':    'PSL',       'lrpdc':    'PDC',
    'lrpds':    'PDS',       'lrppr':    'PPR',       'lrptn':    'PTN',
    'lrsd':     'SD',        'lrcid':    'CIDADANIA',
}

# Aproximações ideológicas para partidos ausentes do BLS
_PROXIES_BLS = {
    'ARENA':         ['PDS'],           # ARENA virou PDS em 1979
    'PPB':           ['PP'],            # PPB = PP (1995-2003)
    'AVANTE':        ['PDT'],           # PTdoB/Avante, esquerda similar ao PDT
    'UNIÃO':         ['DEM', 'PSL'],    # fusão DEM + PSL (2021)
    'PR':            ['PL'],            # PR = fusão PL + PRONA (2006)
    'MISSÃO':        ['NOVO', 'PSL'],   # derivação MBL, direita liberal
    'DEM':           ['DEM', 'PFL'],    # DEM = PFL até 2007
    'PFL':           ['PFL', 'DEM'],    # PFL = DEM a partir de 2007

    'PHS':           ['PROS', 'PSD'],
    'PRD':           ['PL',   'PP'],
    'PST':           ['PTB'],
    'PMN':           ['PMDB/MDB'],
    'PSDC':          ['PSC',  'PP'],

    'SDD':           None,
    'SOLIDARIEDADE': None,
    'SOLID':         None,
    'PTC':           None,
    'AGIR':          None,
}


# Definindo e padronizando atores políticos
INSTITUCIONAIS = {
    'Executivo', 'Comissão', 'Legislativo', 'Judiciário', 'TCU', 'MPU', 
    'Iniciativa Popular', 'Popular; Comissão', 'Sem partido', '—', '',
}

def _norm_partido(p):
    p = str(p).strip().replace('*', '').strip()
    if not p or p in INSTITUCIONAIS:
        return None
    if p in ('S. PART.', 'S.PART.', 'SEM PARTIDO', 'Sem partido'):
        return '0 - Sem Partido'
    if p in ('Popular', 'POPULAR'):
        return '0 - Popular'
    if p in ('PODE', 'PODEMOS'):
        return 'PODEMOS'
    if p in ('MDB', 'PMDB'):
        return 'PMDB/MDB'
    if any(x in p for x in ('MISTA', 'CPI')):
        return None
    if len(p) > 12:
        return None
    return p


def _carrega_scores_bls():
    df_bls = pd.read_csv(_BLS_PATH).replace(-999, np.nan)
    scores = (df_bls
              .groupby('wave')[list(_LR_COLS.keys())]
              .mean()
              .reset_index())
    return (scores
            .melt(id_vars='wave', var_name='col', value_name='score_lr')
            .assign(partido=lambda x: x['col'].map(_LR_COLS))
            .dropna(subset=['score_lr'])
            .groupby(['wave', 'partido'])['score_lr']
            .mean()
            .round(2)
            .reset_index())


def _extrai_partido_bls(tipo_autor):
    if pd.isna(tipo_autor) or tipo_autor in ('—', ''):
        return None
    for t in str(tipo_autor).split(';'):
        t = t.strip()
        if t and t not in INSTITUCIONAIS and len(t) <= 12:
            return _norm_partido(t)
    return None


def _espectro(s):
    if pd.isna(s):  return 'Não identificado'
    if s <= 3.0:    return 'Esquerda'
    if s <= 4.5:    return 'Centro-Esquerda'
    if s <= 5.5:    return 'Centro'
    if s <= 7.0:    return 'Centro-Direita'
    return                 'Direita'


def enriquecer_bls(df, ano_inicio, ano_fim):
    saida       = DIR_OUT / f'etapa4_final_{ano_inicio}_{ano_fim}.parquet'
    scores_long = _carrega_scores_bls()
    waves       = sorted(scores_long['wave'].unique())
    scores_idx  = scores_long.set_index(['wave', 'partido'])['score_lr']

    # waves ordenadas por proximidade a um ano (com suporte a empates)
    def _waves_por_dist(ano):
        ano = int(ano)
        return sorted(waves, key=lambda w: abs(w - ano))

    def _score_partido(partido, ano):
        '''
        Busca score com:
        1. Resolução de proxies
        2. Wave mais próxima com empate → média
        3. Fallback para wave mais próxima com dados disponíveis
        '''
        partidos = _PROXIES_BLS.get(partido, [partido])
        if partidos is None:
            return np.nan

        ano_int       = int(ano)
        waves_ord     = _waves_por_dist(ano_int)
        min_dist      = abs(waves_ord[0] - ano_int)
        nearest       = [w for w in waves_ord if abs(w - ano_int) == min_dist]
        fallback_rest = [w for w in waves_ord if abs(w - ano_int) > min_dist]

        scores = []
        for p in partidos:
            # tenta waves mais próximas (com empate)
            s_near = []
            for w in nearest:
                try:
                    s = scores_idx.loc[(w, p)]
                    if not pd.isna(s):
                        s_near.append(s)
                except KeyError:
                    pass

            if s_near:
                scores.extend(s_near)
            else:
                # fallback: wave mais próxima que tenha dado
                for w in fallback_rest:
                    try:
                        s = scores_idx.loc[(w, p)]
                        if not pd.isna(s):
                            scores.append(s)
                            break
                    except KeyError:
                        pass

        return round(float(np.mean(scores)), 2) if scores else np.nan

    print('  Extraindo partidos...')
    df['partido_bls'] = df['Tipo/Partido Autor'].apply(_extrai_partido_bls)

    print('  Calculando scores BLS...')
    df['score_lr'] = df.apply(
        lambda r: _score_partido(r['partido_bls'], r['Ano'])
        if r['partido_bls'] is not None else np.nan,
        axis=1
    )
    df['espectro'] = df['score_lr'].apply(_espectro)

    df.to_parquet(saida, index=False)

    print(f'\nArquivo: {saida}')
    print(f'Com score BLS: {df["score_lr"].notna().sum()} / {len(df)}')
    print()
    print(df['espectro'].value_counts().to_string())


# == PIPELINE PRINCIPAL ========================================================

COLUNAS = {
    'origem':             'Origem',
    'tipo':               'Tipo de Proposta',
    'numero':             'Número',
    'ano':                'Ano',
    'data_apre':          'Data de Apresentação',
    'parlamentar':        'Parlamentar',
    'tipo_autor':         'Tipo/Partido Autor',
    'uf':                 'UF',
    'ementa':             'Ementa',
    'termo_busca':        'Termo de Busca',
    'tramitando':         'Tramitando',
    'id_interno':         'ID Interno',
    'id_processo':        'ID Processo',
    'uri':                'URI',
}

FILL_VAZIO = {
    'ID Processo': '—',
    'Tramitando':  '—',
    'UF':          '—',
}

# ── Pastas ─────────────────────────────────────────────────────────────────────
DIR_FILT = Path('data/filtered')
DIR_OUT  = Path('data/output')

for _d in [DIR_FILT, DIR_OUT]:
    _d.mkdir(parents=True, exist_ok=True)


def _consolida_e_salva(ano_inicio, ano_fim):
    '''Lê todos os parquets anuais, consolida e salva o output final.'''

    arquivos_cd   = sorted(DIR_FILT.glob('camara_????.parquet'))
    arquivos_sfcn = sorted(DIR_FILT.glob('senado_????.parquet'))

    if not arquivos_cd and not arquivos_sfcn:
        print('Nenhum arquivo encontrado para consolidar.')
        return

    df_cd   = pd.concat([pd.read_parquet(f) for f in arquivos_cd],
                        ignore_index=True) if arquivos_cd   else pd.DataFrame()
    df_sfcn = pd.concat([pd.read_parquet(f) for f in arquivos_sfcn],
                        ignore_index=True) if arquivos_sfcn else pd.DataFrame()

    df_renamed = pd.concat([df_cd, df_sfcn], ignore_index=True).rename(columns=COLUNAS)
    df_renamed = df_renamed[~df_renamed['assunto_especifico'].isin(ASSUNTOS_EXCLUIR)]

    df = (df_renamed
          [[c for c in COLUNAS.values() if c in df_renamed.columns]]
          .assign(**{'Tipo de Proposta': lambda x:
                     x['Tipo de Proposta'].map(TIPO_NORM).fillna(x['Tipo de Proposta'])})
          .loc[lambda x: ~x['Ementa'].str.contains(EXCLUIR_EMENTA, na=False)]
          .fillna(FILL_VAZIO)
          .sort_values(['Origem', 'Ano', 'Tipo de Proposta'])
          .reset_index(drop=True))

    df['Data de Apresentação'] = (pd.to_datetime(df['Data de Apresentação'],
                                                 errors='coerce')
                                    .dt.strftime('%Y-%m-%d'))

    df['Número'] = (df['Número']
                    .str.extract(r'(\d+)/\d{4}$', expand=False)
                    .fillna(df['Número']))

    # remove duplicatas que podem surgir se um ano foi reprocessado
    df = df.drop_duplicates(subset='ID Interno').reset_index(drop=True)

    for col in ['Número', 'ID Interno', 'ID Processo', 'Ano']:
        df[col] = df[col].astype(str)

    saida = DIR_OUT / f'etapa1_output_{ano_inicio}_{ano_fim}.parquet'
    df.to_parquet(saida, index=False)

    print(f'\nArquivo: {saida}')
    print(f'Total: {len(df)} registros '
          f'({len(df[df["Origem"]=="CD"])} proposições CD · '
          f'{len(df[df["Origem"].isin(["SF","CN"])])} matérias SF/CN)\n')

    print(pd.crosstab(df['Tipo de Proposta'], df['Origem'],
                      margins=True, margins_name='Total').to_string())


if __name__ == '__main__':

    for ano in range(ANO_INICIO, ANO_FIM + 1):
        arq_cd   = DIR_FILT / f'camara_{ano}.parquet'
        arq_sfcn = DIR_FILT / f'senado_{ano}.parquet'

        if arq_cd.exists() and arq_sfcn.exists():
            print(f'  {ano} → já processado, pulando')
            continue

        print(f'\n{"="*60}')
        print(f'  Ano: {ano}')
        print(f'{"="*60}')

        if not arq_cd.exists():
            print('--- Câmara ---')
            df_cd_ano = coletar_camara(TERMOS, TIPOS_CAMARA, ano, ano)
            df_cd_ano.to_parquet(arq_cd, index=False)
            print(f'  → {len(df_cd_ano)} proposições · salvo em {arq_cd.name}')
        else:
            print(f'  Câmara {ano} → já existe, pulando')

        if not arq_sfcn.exists():
            print('--- Senado / CN ---')
            df_sfcn_ano = coletar_senado(TERMOS, TIPOS_SENADO, ano, ano)
            df_sfcn_ano.to_parquet(arq_sfcn, index=False)
            print(f'  → {len(df_sfcn_ano)} matérias · salvo em {arq_sfcn.name}')
        else:
            print(f'  Senado {ano} → já existe, pulando')

    print(f'\n{"="*60}')
    print('  Consolidando todos os anos')
    print(f'{"="*60}')
    
    _consolida_e_salva(ANO_INICIO, ANO_FIM)

    # ── Enriquecimento de status ───────────────────────────────────────────────
    saida_enr = DIR_OUT / f'etapa2_enriquecido_{ANO_INICIO}_{ANO_FIM}.parquet'

    if saida_enr.exists():
        print(f'\n  Arquivo enriquecido já existe, pulando')
    else:
        print(f'\n{"="*60}')
        print('  Enriquecendo status legislativo')
        print(f'{"="*60}')

        df_base = pd.read_parquet(DIR_OUT / f'etapa1_output_{ANO_INICIO}_{ANO_FIM}.parquet')
        df_enr  = enriquecer_status(df_base)
        df_enr['situacao'] = df_enr['situacao'].astype(str).replace('None', '')
        df_enr.to_parquet(saida_enr, index=False)

        print(f'\nArquivo: {saida_enr}')
        print(df_enr['situacao_norm'].value_counts())

    # ── Avaliação LLM ─────────────────────────────────────────────────────────
    saida_av = DIR_OUT / f'etapa3_avaliado_{ANO_INICIO}_{ANO_FIM}.parquet'

    if saida_av.exists():
        _n_pend = pd.to_numeric(
            pd.read_parquet(saida_av)['llm_nota'], errors='coerce'
        ).isna().sum()
        if _n_pend == 0:
            print('\n  Avaliação LLM já completa, pulando.')
        else:
            print(f'\n{"="*60}')
            print('  Avaliação LLM')
            print(f'{"="*60}')
            df_enr = pd.read_parquet(saida_enr)
            avaliar_llm(df_enr, ANO_INICIO, ANO_FIM)
    else:
        print(f'\n{"="*60}')
        print('  Avaliação LLM')
        print(f'{"="*60}')
        df_enr = pd.read_parquet(saida_enr)
        avaliar_llm(df_enr, ANO_INICIO, ANO_FIM)

    # ── Enriquecimento ideológico BLS ─────────────────────────────────────────
    if not saida_av.exists():
        print('\n  Base avaliada por LLM não encontrada. Rode o LLM primeiro.')
    elif (DIR_OUT / f'etapa4_final_{ANO_INICIO}_{ANO_FIM}.parquet').exists():
        print('\n  Base final já existe, pulando.')
    elif not _BLS_PATH.exists():
        print(f'\n  {_BLS_PATH} não encontrado.')
    else:
        print(f'\n{"="*60}')
        print('  Enriquecimento Ideológico (BLS)')
        print(f'{"="*60}')
        df_av = pd.read_parquet(saida_av)
        enriquecer_bls(df_av, ANO_INICIO, ANO_FIM)