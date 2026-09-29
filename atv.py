import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RISCO_DICT = {
    'C100': 'Baixo',
    'C101': 'Alto',
    'C102': 'Comum',
    'C103': 'Alterado',
    'C104': 'Fora da media',
}


def zscore_numpy(valores: np.ndarray, grupos: np.ndarray) -> np.ndarray:
    _, inv = np.unique(grupos, return_inverse=True)
    n = np.bincount(inv)
    media = np.bincount(inv, weights=valores) / n
    var = np.bincount(inv, weights=(valores - media[inv]) ** 2) / (n - 1)
    return (valores - media[inv]) / np.sqrt(var)[inv]


def main(caminho='transacoes.csv', mostrar=False):
    df = pd.read_csv(caminho, encoding='latin1')

    medianas = df.groupby('estado_cliente')['valor'].median()
    df['valor'] = df['valor'].fillna(df['estado_cliente'].map(medianas))

    df['plataforma'] = 'Mobile'

    df['data_transacao'] = (
        pd.to_datetime(df['data_transacao'])
        .astype('datetime64[ns]')
        .dt.tz_localize('America/Sao_Paulo')
    )
    dt = df['data_transacao'].dt
    df['dia_semana'] = dt.day_name()
    df['mes'] = dt.month

    df = df.drop_duplicates(keep='first').reset_index(drop=True)

    mes = df['mes'].to_numpy()
    uf = df['estado_cliente'].to_numpy()
    val = df['valor'].to_numpy()
    mascara = (mes == 9) & ((uf == 'SP') | (uf == 'RJ')) & (val > 5000)
    df_setembro = df[mascara]

    df['nivel_risco'] = df['id_cliente'].map(RISCO_DICT)
    tabela = pd.pivot_table(
        df, values='valor', index='mes', columns='nivel_risco',
        aggfunc='sum', margins=True,
    )

    df['zscore'] = zscore_numpy(val.astype(float), uf)
    df_anomalias = df[df['zscore'].to_numpy() > 2.5]

    df['data_dia'] = df['data_transacao'].dt.date
    serie_diaria = df.groupby('data_dia')['valor'].sum()
    media_movel = serie_diaria.rolling(window=7).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(serie_diaria))
    ax.plot(x, serie_diaria.to_numpy(), marker='o', label='Valor diario')
    ax.plot(x, media_movel.to_numpy(), linewidth=2.5,
            label='Media movel (7 dias)')
    passo = max(1, len(x) // 10)
    ax.set_xticks(x[::passo])
    ax.set_xticklabels([str(d) for d in serie_diaria.index[::passo]],
                       rotation=45, ha='right')
    ax.set_ylim(0)
    ax.set_title('Transacoes diarias e media movel de 7 dias')
    ax.set_xlabel('Data')
    ax.set_ylabel('Valor total (R$)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if mostrar:
        plt.show()

    return dict(df=df, df_setembro=df_setembro, tabela=tabela,
                df_anomalias=df_anomalias, serie_diaria=serie_diaria,
                media_movel=media_movel, fig=fig)


if __name__ == '__main__':
    main(mostrar=True)