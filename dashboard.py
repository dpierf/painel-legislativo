# == 0. Importações =============================================================

import re
import dash
import requests
import pandas                       as pd
import plotly.express               as px
import plotly.graph_objects         as go
import dash_bootstrap_components    as dbc
from dash                           import dcc, html, Input, Output


# == 1. Dados & Pré-processamento ===============================================

df = pd.read_parquet('data/output/etapa4_final_1988_2025.parquet')
df = df[df['llm_decisao'] == 'incluir'].reset_index(drop=True)
df['llm_nota'] = pd.to_numeric(df['llm_nota'], errors='coerce')
df['score_lr']  = pd.to_numeric(df['score_lr'],  errors='coerce')

# ── Mapeamentos ────────────────────────────────────────────────────────────────

ORIGEM_LABEL = {
    'CD': 'Câmara dos Deputados',
    'SF': 'Senado Federal',
    'CN': 'Congresso Nacional',
}

TIPO_LABEL = {
    'PL':  'Projeto de Lei',
    'PLP': 'Projeto de Lei Complementar',
    'PEC': 'Proposta de Emenda Constitucional',
    'MPV': 'Medida Provisória',
    'PDL': 'Projeto de Decreto Legislativo',
    'PLV': 'Projeto de Lei de Conversão',
    'PLS': 'Projeto de Lei do Senado',
    'PLC': 'Projeto de Lei da Câmara (SF)',
    'PLN': 'Projeto de Lei do Congresso',
    'PRC': 'Projeto de Resolução',
    'PRN': 'Projeto de Resolução do CN',
    'MSG': 'Mensagem',
    'PFC': 'Proposta de Fiscalização e Controle',
    'SUG': 'Sugestão (Iniciativa Popular)',
}

UF_REGIAO = {
    'AC':'Norte',   'AM':'Norte',   'AP':'Norte',  'PA':'Norte',
    'RO':'Norte',   'RR':'Norte',   'TO':'Norte',
    'AL':'Nordeste','BA':'Nordeste','CE':'Nordeste','MA':'Nordeste',
    'PB':'Nordeste','PE':'Nordeste','PI':'Nordeste','RN':'Nordeste',
    'SE':'Nordeste',
    'DF':'Centro-Oeste','GO':'Centro-Oeste','MS':'Centro-Oeste','MT':'Centro-Oeste',
    'ES':'Sudeste', 'MG':'Sudeste', 'RJ':'Sudeste', 'SP':'Sudeste',
    'PR':'Sul',     'RS':'Sul',     'SC':'Sul',
}

ORDEM_LEG = [
    '48ª (1987-1991)', '49ª (1991-1995)', '50ª (1995-1999)',
    '51ª (1999-2003)', '52ª (2003-2007)', '53ª (2007-2011)',
    '54ª (2011-2015)', '55ª (2015-2019)', '56ª (2019-2023)',
    '57ª (2023-2027)',
]

GRUPO_TIPO = {
    'PL':  'Projetos de Lei',  'PLP': 'Projetos de Lei',
    'PLS': 'Projetos de Lei',  'PLC': 'Projetos de Lei',
    'PLN': 'Projetos de Lei',
    'PEC': 'Emendas Constitucionais',
    'MPV': 'Medidas Provisórias', 'PLV': 'Medidas Provisórias',
    'PDL': 'Decretos Legislativos',
    'PRC': 'Outros', 'PRN': 'Outros', 'MSG': 'Outros',
    'PFC': 'Outros', 'SUG': 'Outros',
}

def _leg(ano):
    ano = int(ano)
    if   ano <= 1990: return '48ª (1987-1991)'
    elif ano <= 1994: return '49ª (1991-1995)'
    elif ano <= 1998: return '50ª (1995-1999)'
    elif ano <= 2002: return '51ª (1999-2003)'
    elif ano <= 2006: return '52ª (2003-2007)'
    elif ano <= 2010: return '53ª (2007-2011)'
    elif ano <= 2014: return '54ª (2011-2015)'
    elif ano <= 2018: return '55ª (2015-2019)'
    elif ano <= 2022: return '56ª (2019-2023)'
    else:             return '57ª (2023-2027)'

INSTITUCIONAIS = {
    'Executivo', 'Comissão', 'Legislativo', 'Judiciário',
    'TCU', 'MPU', 'Iniciativa Popular', 'Popular; Comissão',
    'Sem partido', '—', '',
}

def _norm_partido(p):
    p = str(p).strip().replace('*', '').strip()
    if not p or p in INSTITUCIONAIS:
        return None
    if p in ('S. PART.', 'S.PART.', 'SEM PARTIDO', 'Sem partido', 'S/PART'):
        return '0 - Sem Partido'
    if p in ('Popular', 'POPULAR', 'Iniciativa Popular'):
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

_MINUSCULAS = {'de','da','do','das','dos','e','em','com','por','para',
               'a','o','as','os','no','na','nos','nas'}

def _capitaliza_termo(t):
    words = str(t).split()
    return ' '.join(
        w.capitalize() if i == 0 or w.lower() not in _MINUSCULAS else w.lower()
        for i, w in enumerate(words)
    )

# ── Enriquecimento do DataFrame ────────────────────────────────────────────────

df['Origem Extenso'] = df['Origem'].map(ORIGEM_LABEL).fillna(df['Origem'])
df['Tipo Extenso']   = df['Tipo de Proposta'].map(TIPO_LABEL).fillna(df['Tipo de Proposta'])
df['Legislatura']    = df['Ano'].apply(_leg)
df['Região']         = df['UF'].map(UF_REGIAO).fillna('Não identificada')
df['Ano_int']        = df['Ano'].astype(int)

def _e_parlamentar(tipo_autor):
    if not tipo_autor or pd.isna(tipo_autor) or tipo_autor in ('—', ''):
        return False
    for t in str(tipo_autor).split(';'):
        t = t.strip()
        if t and t not in INSTITUCIONAIS and len(t) <= 12:
            return True
    return False

df['é_parlamentar'] = df['Tipo/Partido Autor'].apply(_e_parlamentar)

# ── Opções dos filtros ─────────────────────────────────────────────────────────

anos      = sorted(df['Ano_int'].unique())
origens   = sorted(df['Origem Extenso'].unique())
grupos    = ['Projetos de Lei', 'Emendas Constitucionais', 'Medidas Provisórias', 'Decretos Legislativos', 'Outros']
tipos     = sorted(df['Tipo Extenso'].dropna().unique())
regioes   = ['Norte','Nordeste','Centro-Oeste','Sudeste','Sul','Não identificada']
espectros = ['Esquerda', 'Centro-Esquerda', 'Centro', 'Centro-Direita', 'Direita', 'Não identificado']
ufs = sorted([
    u.strip()
    for u in df['UF'].dropna().str.split(';').explode()
    if u.strip() and u.strip() not in ('—', '')
    and len(u.strip()) == 2  # garante só siglas válidas
])
ufs = sorted(set(ufs))
partidos = sorted(set(filter(None, [
    _norm_partido(p.strip())
    for p in df['Tipo/Partido Autor'].dropna().str.split(';').explode()
])))

politico = sorted(set([
    ' '.join(
        w.capitalize() if w.upper() not in {m.upper() for m in _MINUSCULAS}
        else w.lower()
        for w in re.sub(r'\([^)]*\)', '', nome.strip().upper()).split()
    ).strip()
    for p in df['Parlamentar'].dropna()
    for nome in re.split(r'[;,]', p)
    if nome.strip() and nome.strip() not in ('—', '')
]))
politico = [p for p in politico if len(p) > 2]


termos = [
    {'label': _capitaliza_termo(t), 'value': t}
    for t in sorted(df['Termo de Busca'].dropna().unique())
]

# ── GeoJSON estados do Brasil ──────────────────────────────────────────────────

_GEOJSON_URL = (
    'https://raw.githubusercontent.com/codeforamerica/'
    'click_that_hood/master/public/data/brazil-states.geojson'
)
try:
    _geojson = requests.get(_GEOJSON_URL, timeout=10).json()
    for f in _geojson['features']:
        f['id'] = f['properties']['sigla']
    GEOJSON_OK = True
except Exception:
    GEOJSON_OK = False


# == 2. App =====================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title='Agenda Legislativa | Pobreza & Desigualdade',
)

app.index_string = app.index_string.replace(
    '</head>',
    '<style>.rc-slider-mark-text { color: #aaaaaa !important; }</style></head>'
)

server = app.server

TEMPLATE = 'plotly_dark'
BG       = 'rgba(0,0,0,0)'
COR_CD   = '#4e9af1'
COR_SF   = '#f1a94e'
COR_CN   = '#7ed17e'

PALETA   = px.colors.qualitative.Bold
PALETA_TERMOS = [ #Específica para as categorias das iniciativas
    '#4e9af1', '#f1a94e', '#7ed17e', '#e05c5c', '#b57bee',
    '#f1e24e', '#4ecdf1', '#f14e9a', '#a0d4a0', '#f17a4e',
    '#4ef1c8', '#c8f14e',
]

CORES_ESPECTRO = {
    'Esquerda':          '#c62828',
    'Centro-Esquerda':   '#ef9a9a',
    'Centro':            '#9e9e9e',
    'Centro-Direita':    '#90caf9',
    'Direita':           '#1565c0',
    'Não identificado':  '#444444',
}


# == 3. Layout =================================================================

def _dropdown(label, id_, options, multi=True):
    if options and isinstance(options[0], str):
        opts = [{'label': o, 'value': o} for o in options]
    else:
        opts = options
    return html.Div([
        html.Label(label, style={'color': '#aaaaaa', 'fontSize': '11px',
                                 'marginBottom': '4px', 'display': 'block'}),
        dcc.Dropdown(id=id_, options=opts, multi=multi, clearable=True,
                     style={'fontSize': '12px'}),
    ], className='mb-3')


sidebar = dbc.Col([
    html.H6('FILTROS', 
            style={
                'color': '#aaaaaa',
                'fontWeight': 'bold',
                'letterSpacing': '2px',
                'fontSize': '11px',
                'marginBottom': '16px',
                'marginTop': '8px',
                }, className='text-uppercase text-muted fw-bold mb-3 mt-2'),
    _dropdown('Legislatura',            'f-leg',        ORDEM_LEG),
    _dropdown('Origem',                 'f-origem',     origens),
    _dropdown('Grupo de Proposta',      'f-grupo',      grupos),
    _dropdown('Tipo de Proposta',       'f-tipo',       tipos),
    _dropdown('Termo de Busca',         'f-termo',      termos),
    _dropdown('Região',                 'f-regiao',     regioes),
    _dropdown('UF',                     'f-uf',         ufs),
    _dropdown('Tipo de Autor', 'f-autor', [
        'Políticos', 'Executivo', 'Comissão', 'Legislativo',
        'Judiciário', 'TCU', 'MPU', 'Iniciativa Popular',
    ]   ),
    _dropdown('Espectro Político',      'f-espectro',   espectros),
    _dropdown('Partido',                'f-partido',    partidos),
    _dropdown('Político',               'f-politico',   politico),

    html.Hr(className='border-secondary'),
    html.Label('Período (ano)', className='text-muted small mb-1'),
    dcc.RangeSlider(
        id='f-ano',
        min=anos[0], max=anos[-1], step=1,
        value=[anos[0], anos[-1]],
        marks={int(a): str(a) for a in anos[::4]},
        tooltip={'placement': 'bottom', 'always_visible': False},
    ),
], width=2, className='p-3',
   style={
       'backgroundColor': '#111111',
       'borderRight': '1px solid #333333',
       'minHeight': '100vh',
       'overflowY': 'auto',
       'position': 'sticky',
       'top': 0,
   })


def _kpi(id_, label):
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.P(label, className='mb-1',
                   style={'fontSize': '11px', 'textTransform': 'uppercase',
                          'letterSpacing': '0.5px', 'color': '#aaaaaa'}),
            html.H3(id=id_, className='fw-bold mb-0',
                    style={'color': '#ffffff'}),
        ], className='p-3'),
        style={'backgroundColor': '#1a1a1a', 'border': '1px solid #333333'}),
    width=3, className='mb-3')


def _barra_casa(pct_cd, pct_sf, pct_cn):
    def _seg(cor, pct, label):
        return html.Div([
            html.Div(style={'backgroundColor': cor, 'height': '8px',
                            'borderRadius': '2px', 'marginBottom': '4px'}),
            html.Span(f'{label} {pct:.0f}%',
                      style={'color': '#aaaaaa', 'fontSize': '10px',
                             'whiteSpace': 'nowrap'}),  # ← evita quebra
        ], style={'width': f'{max(pct, 12):.0f}%',   # ← mínimo maior
                  'paddingRight': '6px'})

    return dbc.Card(dbc.CardBody([
        html.P('Distribuição por Casa', className='mb-2',
               style={'fontSize': '11px', 'textTransform': 'uppercase',
                      'letterSpacing': '0.5px', 'color': '#aaaaaa'}),
        html.Div([_seg(COR_CD, pct_cd, 'CD'),
                  _seg(COR_SF, pct_sf, 'SF'),
                  _seg(COR_CN, pct_cn, 'CN')],
                 style={'display': 'flex', 'width': '100%',
                        'alignItems': 'flex-start',
                        'overflow': 'hidden'}),  # ← contém tudo na linha
    ], className='p-3', style={'minHeight': '100px'}),
    style={'backgroundColor': '#1a1a1a', 'border': '1px solid #333333', 'height': '100%'})


kpis = dbc.Row([
    _kpi('k-total',        'Proposições no período'),
    _kpi('k-media',        'Média por ano'),
    dbc.Col(html.Div(id='k-casa'), width=3, className='mb-3'),
    _kpi('k-promulgadas',  'Promulgadas'),
    _kpi('k-parlamentares','Parlamentares únicos'),
    _kpi('k-por-parl',     'Proposições / parlamentar'),
    _kpi('k-popular',      'Iniciativa popular'),
    _kpi('k-mpv',          'Medidas Provisórias'),
], className='g-3 mb-2')


graficos = html.Div([
    dbc.Row([
        dbc.Col(dcc.Graph(id='g-barras-leg'),   width=6),
        dbc.Col(dcc.Graph(id='g-linhas-ano'),   width=6),
    ], className='mb-3'),
    dbc.Row([                                              
        dbc.Col(dcc.Graph(id='g-fig12'),        width=6),
        dbc.Col(dcc.Graph(id='g-fig13'),        width=6),
    ], className='mb-3'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='g-area-tipo'),    width=6),
        dbc.Col(dcc.Graph(id='g-origem-autor'), width=6),
    ], className='mb-3'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='g-treemap'),      width=6),
        dbc.Col(dcc.Graph(id='g-mapa'),         width=6),
    ], className='mb-3'),
    dbc.Row([                                              
        dbc.Col(dcc.Graph(id='g-fig14'),        width=6),
        dbc.Col(dcc.Graph(id='g-fig15'),        width=6),
    ], className='mb-3'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='g-termos-leg'),   width=6),
        dbc.Col(dcc.Graph(id='g-ranking-parl'), width=6),
    ], className='mb-3'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='g-matriz'),       width=12),
    ], className='mb-3'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='g-sankey'),       width=12),
    ], className='mb-3'),
        dbc.Row([
        dbc.Col(dcc.Graph(id='g-tabela'), width=12),
    ], className='mb-3'),
])


app.layout = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.Div([
        html.H4('Agenda Legislativa · Pobreza & Desigualdade',
                className='fw-bold mb-0', style={'color': '#ffffff'}),
        html.Small('Brasil · 1988–2025', style={'color': '#888888'}),
    ], className='py-3 px-4'),
    style={'backgroundColor': '#111111', 'borderBottom': '1px solid #333333'})),
    dbc.Row([
        sidebar,
        dbc.Col([
            html.Div(kpis,    className='mt-4'),
            html.Div(graficos),
        ], width=10, className='p-4'),
    ]),
], style={'backgroundColor': '#0d0d0d', 'minHeight': '100vh'})


# == 4. Callbacks ==============================================================

INPUTS = [
    Input('f-origem',   'value'),
    Input('f-grupo',    'value'),
    Input('f-tipo',     'value'),
    Input('f-leg',      'value'),
    Input('f-autor',    'value'),
    Input('f-espectro', 'value'),
    Input('f-partido',  'value'),
    Input('f-politico', 'value'),
    Input('f-regiao',   'value'),
    Input('f-uf',       'value'),
    Input('f-termo',    'value'),
    Input('f-ano',      'value'),
]

def _cat_autor(ta):
    if pd.isna(ta) or ta in ('—', ''):
        return 'Não identificado'
    for t in str(ta).split(';'):
        t = t.strip()
        if t in INSTITUCIONAIS:
            return t if t not in ('', '—') else 'Não identificado'
        if t and len(t) <= 12:
            return 'Políticos'
    return 'Não identificado'

def _filtrar(origem, grupo, tipo, leg, autor, espectro, partido, politico, regiao, uf, termo, ano):
    d = df.copy()
    if origem:      d = d[d['Origem Extenso'].isin(origem)]
    if grupo:       d = d[d['Tipo de Proposta'].map(GRUPO_TIPO).isin(grupo)]
    if tipo:        d = d[d['Tipo Extenso'].isin(tipo)]
    if leg:         d = d[d['Legislatura'].isin(leg)]
    if regiao:      d = d[d['Região'].isin(regiao)]
    if uf:
        mask = d['UF'].apply(
            lambda x: any(u.strip() in uf for u in str(x).split(';'))
        )
        d = d[mask]
    if termo:       d = d[d['Termo de Busca'].isin(termo)]
    if autor:       d = d[d['Tipo/Partido Autor'].apply(lambda x: _cat_autor(str(x)) in autor)]
    if espectro:    d = d[d['espectro'].isin(espectro)]
    if partido:
        mask = d['Tipo/Partido Autor'].apply(
            lambda x: any(
                _norm_partido(p.strip()) in partido
                for p in str(x).split(';')
            )
        )
        d = d[mask]
    if politico:
        def _norm_nome(nome):
            return ' '.join(
                w.capitalize() if w.upper() not in {m.upper() for m in _MINUSCULAS}
                else w.lower()
                for w in re.sub(r'\([^)]*\)', '', nome.strip().upper()).split()
            ).strip()

        mask = d['Parlamentar'].apply(
            lambda x: any(
                _norm_nome(n) in politico
                for n in re.split(r'[;,]', str(x))
            )
        )
        d = d[mask]
    if ano:         d = d[(d['Ano_int'] >= ano[0]) & (d['Ano_int'] <= ano[1])]
    return d


def _layout(title='', height=450):
    return dict(
        template=TEMPLATE,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text=f'<b>{title}</b>', font=dict(size=12), x=0, xref='paper'),
        font=dict(size=11),
        legend=dict(font=dict(size=10)),
    )


@app.callback(
    [Output('k-total',          'children'),
     Output('k-media',          'children'),
     Output('k-casa',           'children'),
     Output('k-promulgadas',    'children'),
     Output('k-parlamentares',  'children'),
     Output('k-por-parl',       'children'),
     Output('k-popular',        'children'),
     Output('k-mpv',            'children'),
     Output('g-barras-leg',     'figure'),
     Output('g-linhas-ano',     'figure'),
     Output('g-fig12',          'figure'),
     Output('g-fig13',          'figure'),
     Output('g-area-tipo',      'figure'),
     Output('g-origem-autor',   'figure'),
     Output('g-treemap',        'figure'),
     Output('g-mapa',           'figure'),
     Output('g-fig14',          'figure'),
     Output('g-fig15',          'figure'),
     Output('g-termos-leg',     'figure'),
     Output('g-ranking-parl',   'figure'),
     Output('g-matriz',         'figure'),
     Output('g-sankey',         'figure'),
     Output('g-tabela',         'figure'),
     ],
    INPUTS,
)
def update(origem, grupo, tipo, leg, autor, espectro, partido, politico, regiao, uf, termo, ano):
    d = _filtrar(origem, grupo, tipo, leg, autor, espectro, partido, politico, regiao, uf, termo, ano)

    n  = len(d)

    _vazio = go.Figure().update_layout(**_layout('Sem dados para o filtro selecionado'))

    if n == 0:
        return ['—'] * 8 + [_vazio] * 15

    anos_uniq = max(d['Ano_int'].nunique(), 1)
    d_parl    = d[d['é_parlamentar']]

    # ── KPIs ──────────────────────────────────────────────────────────────────

    k_total = f'{n:,}'.replace(',', '.')

    k_media = f'{n / anos_uniq:.1f}'.replace('.', ',')

    casa   = d['Origem'].value_counts(normalize=True).mul(100)
    pct_cd = float(casa.get('CD', 0))
    pct_sf = float(casa.get('SF', 0))
    pct_cn = float(casa.get('CN', 0))
    k_casa = _barra_casa(pct_cd, pct_sf, pct_cn)

    prom   = (d['situacao_norm'] == 'Promulgada').sum()
    k_prom = f'{prom / n:.1%}'

    n_parl  = d_parl['Parlamentar'].nunique()
    k_parl  = f'{n_parl:,}'.replace(',', '.')

    k_por_parl = f'{len(d_parl) / n_parl:.1f}'.replace('.', ',') if n_parl else '—'

    popular = (
        (d['Tipo de Proposta'] == 'SUG') |
        d['Tipo/Partido Autor'].str.contains('Popular', na=False)
    ).sum()
    k_popular = f'{popular / n:.1%}'

    k_mpv = f'{(d["Tipo de Proposta"] == "MPV").sum() / n:.1%}'

    # ── Gráfico 1: Barras por legislatura ─────────────────────────────────────

    leg_ct = (d.groupby('Legislatura').size()
               .reindex(ORDEM_LEG, fill_value=0)
               .reset_index(name='n'))

    fig1 = px.bar(
        leg_ct, x='Legislatura', y='n',
        color_discrete_sequence=[COR_CD],
        labels={'n': '', 'Legislatura': ''},
    ).update_layout(**_layout('Volume por Legislatura'))

    # ── Gráfico 2: Linhas por ano e origem ────────────────────────────────────

    ano_orig = d.groupby(['Ano_int', 'Origem']).size().reset_index(name='n')

    fig2 = px.line(
        ano_orig, x='Ano_int', y='n', color='Origem',
        color_discrete_map={'CD': COR_CD, 'SF': COR_SF, 'CN': COR_CN},
        markers=True,
        labels={'Ano_int': '', 'n': '', 'Origem': ''},
    ).update_layout(**_layout('Proposições por Ano e Casa'))

    # ── Gráfico 3: Área empilhada por tipo e legislatura ─────────────────────

    top5      = d['Tipo de Proposta'].value_counts().head(5).index.tolist()
    d3        = d.copy()
    d3['tp']  = d3['Tipo de Proposta'].where(d3['Tipo de Proposta'].isin(top5), 'Outros')
    tipo_leg  = (d3.groupby(['Legislatura', 'tp'])
                   .size()
                   .reset_index(name='n'))
    tipo_leg['Legislatura'] = pd.Categorical(
        tipo_leg['Legislatura'], categories=ORDEM_LEG, ordered=True
    )

    fig3 = px.area(
        tipo_leg.sort_values('Legislatura'),
        x='Legislatura', y='n', color='tp',
        color_discrete_sequence=PALETA,
        groupnorm='percent',
        labels={'n': '', 'Legislatura': '', 'tp': ''},
    ).update_layout(**_layout('Tipos por Legislatura'))

    # ── Gráfico 4: Origem das proposições ─────────────────────────────────────

    d4 = d.copy()
    d4['cat'] = d4['Tipo/Partido Autor'].apply(_cat_autor)
    cat_ct    = d4['cat'].value_counts().reset_index()
    cat_ct.columns = ['Categoria', 'n']

    fig4 = px.pie(
        cat_ct, values='n', names='Categoria',
        color_discrete_sequence=PALETA, hole=0.45,
    ).update_traces(
        textposition='inside', textinfo='percent+label'
    ).update_layout(**_layout('Origem das Proposições'))

    # ── Gráfico 5: Treemap de partidos ────────────────────────────────────────

    part_ct = (d_parl
               .assign(p_plot=d_parl['Tipo/Partido Autor']
                                  .str.split(';').str[0].str.strip()
                                  .apply(_norm_partido))          # ← normaliza
               .dropna(subset=['p_plot'])                         # ← remove None
               .groupby(['p_plot', 'Parlamentar'])
               .size()
               .reset_index(name='n'))
    part_ct.columns = ['Partido', 'Parlamentar', 'n']
    part_ct = part_ct[~part_ct['Partido'].isin(INSTITUCIONAIS)]

    # normaliza nomes
    part_ct['Parlamentar'] = part_ct['Parlamentar'].str.upper().str.strip().apply(
        lambda x: ' '.join(
            w.capitalize() if w not in {m.upper() for m in _MINUSCULAS}
            else w.lower()
            for w in x.split()
        )
    )

    part_ct = (d_parl
               .assign(p_plot=d_parl['Tipo/Partido Autor']
                                  .str.split(';').str[0].str.strip()
                                  .apply(_norm_partido))
               .dropna(subset=['p_plot'])
               .groupby(['p_plot', 'Parlamentar'])
               .size()
               .reset_index(name='n'))
    part_ct.columns = ['Partido', 'Parlamentar', 'n']
    part_ct = part_ct[~part_ct['Partido'].isin(INSTITUCIONAIS)]

    part_ct['Parlamentar'] = part_ct['Parlamentar'].str.upper().str.strip().apply(
        lambda x: ' '.join(
            w.capitalize() if w not in {m.upper() for m in _MINUSCULAS}
            else w.lower()
            for w in x.split()
        )
    )

    # ── "Outros" nos dois níveis ──────────────────────────────────────────────
    party_totals  = part_ct.groupby('Partido')['n'].sum().sort_values(ascending=False)
    top15_parties = party_totals.head(15).index.tolist()

    rows = []
    for partido in top15_parties:
        df_p   = part_ct[part_ct['Partido'] == partido].sort_values('n', ascending=False)
        top10   = df_p.head(10)
        outros = df_p.iloc[10:]['n'].sum()
        for _, row in top10.iterrows():
            rows.append({'Partido': partido, 'Parlamentar': row['Parlamentar'], 'n': row['n']})
        if outros > 0:
            rows.append({'Partido': partido, 'Parlamentar': 'Outros', 'n': int(outros)})

    outros_party = int(party_totals.iloc[10:].sum())
    if outros_party > 0:
        rows.append({'Partido': 'Outros', 'Parlamentar': '—', 'n': outros_party})

    part_ct_plot = pd.DataFrame(rows)

    fig5 = px.treemap(
        part_ct_plot,
        path=['Partido', 'Parlamentar'],
        values='n',
        color='n',
        color_continuous_scale='Blues',
    ).update_traces(
        maxdepth=2,
        texttemplate='<b>%{label}</b><br>%{value}',
        hovertemplate='<b>%{label}</b><br>Proposições: %{value}<extra></extra>',
        textfont=dict(size=12),
    ).update_layout(**_layout('Top 10 Partidos por Volume'))

    # ── Gráfico 6: Mapa ────────────────────────────────────────────────────────

    uf_ct = (d[d['UF'].notna() & ~d['UF'].isin(['—', ''])]
             ['UF'].value_counts().reset_index())
    uf_ct.columns = ['UF', 'n']

    if GEOJSON_OK and len(uf_ct):
        fig6 = px.choropleth(
            uf_ct, geojson=_geojson,
            locations='UF', featureidkey='id',
            color='n', color_continuous_scale='Blues',
        ).update_geos(
            fitbounds='locations',
            visible=False,
            bgcolor='rgba(0,0,0,0)',
        ).update_layout(**_layout('Proposições por UF', height=450))
    else:
        fig6 = px.bar(
            uf_ct.head(15), x='UF', y='n',
            color_discrete_sequence=[COR_CD],
            labels={'n': '', 'UF': ''},
        ).update_layout(**_layout('Proposições por UF (top 15)'))

    # ── Gráfico 7: Termos por legislatura (100% empilhado) ────────────────────

    term_leg = (d.groupby(['Legislatura', 'Termo de Busca'])
                 .size().reset_index(name='n'))
    term_leg['Legislatura'] = pd.Categorical(
        term_leg['Legislatura'], categories=ORDEM_LEG, ordered=True
    )
    term_leg['Termo de Busca'] = term_leg['Termo de Busca'].apply(_capitaliza_termo)


    fig7 = px.bar(
        term_leg.sort_values('Legislatura'),
        x='Legislatura', y='n', color='Termo de Busca',
        barmode='relative',
        color_discrete_sequence=PALETA_TERMOS,
        labels={'n': '%', 'Legislatura': '', 'Termo de Busca': ''},
    ).update_layout(
        barnorm='percent',
        **_layout('Termos de Busca por Legislatura (%)'),)

    # ── Gráfico 8: Ranking de parlamentares ───────────────────────────────────

    rank = (d_parl.groupby('Parlamentar').size()
                  .sort_values(ascending=True).tail(15)
                  .reset_index(name='n'))

    d8 = d_parl.copy()
    d8['parl_norm'] = d8['Parlamentar'].str.upper().str.strip()
    rank = (d8.groupby('parl_norm').size()
              .sort_values(ascending=True).tail(15)
              .reset_index(name='n'))
    rank['parl_norm'] = rank['parl_norm'].apply(
        lambda x: ' '.join(
            w.capitalize() if w not in {m.upper() for m in _MINUSCULAS}
            else w.lower()
            for w in x.split()
        )
    )

    fig8 = px.bar(
        rank, x='n', y='parl_norm', orientation='h',
        color_discrete_sequence=[COR_SF],
        labels={'n': '', 'parl_norm': ''},
    ).update_layout(**_layout('Top 15 Parlamentares', height=450))

    # ── Gráfico 9: Matriz partido × termo ─────────────────────────────────────

    top_p = (d_parl['Tipo/Partido Autor']
             .str.split(';').explode().str.strip()
             .value_counts().head(12).index.tolist())
    top_p = [p for p in top_p if p not in INSTITUCIONAIS]

    d9              = d_parl.copy()
    d9['p_plot']    = d9['Tipo/Partido Autor'].str.split(';').str[0].str.strip()
    d9              = d9[d9['p_plot'].isin(top_p)]

    if len(d9):
        mat = (d9.groupby(['p_plot', 'Termo de Busca'])
                 .size().unstack(fill_value=0))
        mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
        mat = mat[mat.sum(axis=0).sort_values(ascending=False).index]

        fig9 = go.Figure(go.Heatmap(
            z=mat.values, x=mat.columns.tolist(), y=mat.index.tolist(),
            colorscale='Blues',
            hovertemplate='%{y} × %{x}: %{z}<extra></extra>',
        )).update_layout(**_layout('Matriz: Partido × Termo de Busca', height=450))
    else:
        fig9 = _vazio

    # ── Gráfico 10: Sankey partido → tipo ─────────────────────────────────────

    d10 = d_parl.copy()
    d10['p_sk'] = d10['Tipo/Partido Autor'].str.split(';').str[0].str.strip()
    top_p_sk    = d10['p_sk'].value_counts().head(8).index.tolist()
    top_p_sk    = [p for p in top_p_sk if p not in INSTITUCIONAIS]
    top_t_sk    = d10['Tipo de Proposta'].value_counts().head(6).index.tolist()

    d10 = d10[d10['p_sk'].isin(top_p_sk) & d10['Tipo de Proposta'].isin(top_t_sk)]

    if len(d10):
        sk     = (d10.groupby(['p_sk', 'Tipo de Proposta'])
                     .size().reset_index(name='n'))
        tipo_labels = [TIPO_LABEL.get(t, t) for t in top_t_sk]
        nodes  = top_p_sk + tipo_labels
        n_idx_src = {p: i              for i, p in enumerate(top_p_sk)}
        n_idx_tgt = {t: len(top_p_sk) + i for i, t in enumerate(top_t_sk)}

        fig10 = go.Figure(go.Sankey(
            node=dict(
                label=nodes,
                color=[COR_CD] * len(top_p_sk) + [COR_SF] * len(tipo_labels),
                pad=15, thickness=20,
            ),
            link=dict(
                source=[n_idx_src[r['p_sk']]             for _, r in sk.iterrows()],
                target=[n_idx_tgt[r['Tipo de Proposta']] for _, r in sk.iterrows()],
                value=[r['n']                              for _, r in sk.iterrows()],
                color='rgba(255,255,255,0.08)',
            ),
        )).update_layout(**_layout('Sankey: Partido → Tipo de Proposta', height=450))
    else:
        fig10 = _vazio

    # ── Gráfico 11: Tabela de proposições ─────────────────────────────────────

    cols = ['Ano', 'Tipo de Proposta', 'Número', 'Parlamentar',
            'Ementa', 'llm_categoria']
    d11  = (d[[c for c in cols if c in d.columns]]
             .sort_values('Ano', ascending=False)
             .head(500)
             .fillna('—'))

    fig11 = go.Figure(go.Table(
        header=dict(
            values=['Ano', 'Tipo', 'Número', 'Parlamentar', 'Ementa', 'Categoria'],
            fill_color='#1a1a1a',
            font=dict(color='#ffffff', size=11),
            align='left',
            line_color='#333333',
        ),
        cells=dict(
            values=[d11[c].tolist() for c in d11.columns],
            fill_color=[['#111111' if i % 2 == 0 else '#0d0d0d'
                         for i in range(len(d11))]],
            font=dict(color='#cccccc', size=10),
            align=['center', 'center', 'center', 'left', 'left', 'left'],
            height=28,
        ),
    )).update_layout(
        **_layout(f'Proposições — {len(d)} no filtro · exibindo top 500 · ano ↓',
                  height=600)
    )

    # ── Gráfico 12: Nota média por legislatura e tipo ─────────────────────────

    d12 = d.dropna(subset=['score_lr']).copy()
    d12['Grupo'] = d12['Tipo de Proposta'].map(GRUPO_TIPO).fillna('Outros')

    nota_total = (d12.groupby('Legislatura')['score_lr']
                    .mean().reset_index()
                    .assign(Grupo='Total'))
    nota_grupo = d12.groupby(['Legislatura', 'Grupo'])['score_lr'].mean().reset_index()
    nota_leg   = pd.concat([nota_total, nota_grupo], ignore_index=True)
    nota_leg['Legislatura'] = pd.Categorical(nota_leg['Legislatura'], ORDEM_LEG, ordered=True)

    fig12 = px.line(
        nota_leg.sort_values('Legislatura'),
        x='Legislatura', y='score_lr', color='Grupo',
        color_discrete_sequence=['#ffffff'] + PALETA,
        markers=True,
        labels={'score_lr': 'Esquerda ↔ Direita ', 'Legislatura': '', 'Grupo': ''},
    ).update_layout(**_layout('Posição Política Média por Legislatura e Tipo'))

    # ── Gráfico 13: Nota média por legislatura e casa ─────────────────────────

    d13 = d12.copy()
    d13['Casa'] = d13['Origem'].map({'CD': 'Câmara', 'SF': 'Senado', 'CN': 'Senado'})
    nota_casa = (d13.groupby(['Legislatura', 'Casa'])['score_lr']
                    .mean().reset_index())
    nota_casa['Legislatura'] = pd.Categorical(nota_casa['Legislatura'], ORDEM_LEG, ordered=True)

    fig13 = px.line(
        nota_casa.sort_values('Legislatura'),
        x='Legislatura', y='score_lr', color='Casa',
        color_discrete_map={'Câmara': COR_CD, 'Senado': COR_SF},
        markers=True,
        labels={'score_lr': 'Esquerda ↔ Direita ', 'Legislatura': '', 'Casa': ''},
    ).update_layout(**_layout('Posição Política Média por Legislatura e Casa'))

    # ── Gráfico 14: Donut por espectro ────────────────────────────────────────

    esp_ct = (d['espectro'].fillna('Não identificado')
               .value_counts().reset_index())
    esp_ct.columns = ['Espectro', 'n']

    fig14 = px.pie(
        esp_ct, values='n', names='Espectro',
        color='Espectro',
        color_discrete_map=CORES_ESPECTRO,
        hole=0.45,
    ).update_traces(
        textposition='inside', textinfo='percent+label'
    ).update_layout(**_layout('Proposições por Espectro Político'))

    # ── Gráfico 15: Mapa por score ideológico ─────────────────────────────────

    uf_score = (d[d['UF'].notna() & ~d['UF'].isin(['—', '']) &
                  d['score_lr'].notna()]
                .groupby('UF')['score_lr'].mean()
                .reset_index())
    uf_score.columns = ['UF', 'score_lr']

    if GEOJSON_OK and len(uf_score):
        fig15 = px.choropleth(
            uf_score, geojson=_geojson,
            locations='UF', featureidkey='id',
            color='score_lr',
            color_continuous_scale=[[0, '#c62828'], [0.45, '#eeeeee'], [1, '#1565c0']],
            range_color=[1, 10],
        ).update_geos(
            fitbounds='locations', visible=False,
            bgcolor='rgba(0,0,0,0)',
        ).update_layout(
            **_layout('Score Ideológico Médio por UF (1=Esquerda · 10=Direita)'),
            coloraxis_colorbar=dict(title='Score'),
        )
    else:
        fig15 = px.bar(
            uf_score.sort_values('score_lr').head(15),
            x='UF', y='score_lr',
            color='score_lr',
            color_continuous_scale=[[0, '#c62828'], [0.45, '#eeeeee'], [1, '#1565c0']],
            range_color=[1, 10],
        ).update_layout(**_layout('Score Ideológico Médio por UF'))

    return [
        k_total, k_media, k_casa, k_prom,
        k_parl, k_por_parl, k_popular, k_mpv,
        fig1, fig2, 
        fig12, fig13, 
        fig3, fig4,
        fig5, fig6, 
        fig14, fig15,
        fig7, fig8,
        fig9, fig10,
        fig11,
    ]


# == 5. Execução ================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8050)