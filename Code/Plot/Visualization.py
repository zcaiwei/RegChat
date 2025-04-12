import plotly.graph_objects as go
from plotly.colors import qualitative
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import numpy as np


def plot_multilayer_sankey(
    df, 
    gname="network",
    node_colorscale=None,
    path_colors=None,
    title="Multilayer network",
    width=1000,
    height=700,
    font_size=10
):
    """
    Draw multi-layer Sankey diagram to visualize intercellular communication network
    """
    
    nodes = []
    links = {'source': [], 'target': [], 'value': [], 'color': []}
    
    node_index = {}
    current_index = 0
    
    if node_colorscale is None:
        node_colorscale = (
            qualitative.Pastel1 + 
            qualitative.Set3 +
            qualitative.Set2 +
            qualitative.Plotly +
            qualitative.D3 +
            qualitative.Pastel2 +
            qualitative.Light24 +
            qualitative.Alphabet +
            qualitative.Prism +
            qualitative.Antique +
            qualitative.Safe
        )
    
    if path_colors is None:
        path_colors = [
            "rgba(31, 119, 180, 0.5)", "rgba(107, 110, 207, 0.5)", "rgba(128,205,193, 0.5)",
            "rgba(254,224,139, 0.5)", "rgba(148, 103, 189, 0.5)", "rgba(140, 86, 75, 0.5)",
            "rgba(227, 119, 194, 0.5)", "rgba(127, 127, 127, 0.5)", "rgba(188, 189, 34, 0.5)",
            "rgba(23, 190, 207, 0.5)", "rgba(174, 199, 232, 0.5)", "rgba(255, 187, 120, 0.5)",
            "rgba(152, 223, 138, 0.5)", "rgba(255, 152, 150, 0.5)", "rgba(197, 176, 213, 0.5)",
            "rgba(196, 156, 148, 0.5)", "rgba(247, 182, 210, 0.5)", "rgba(199, 199, 199, 0.5)",
            "rgba(219, 219, 141, 0.5)", "rgba(158, 218, 229, 0.5)", "rgba(57, 59, 121, 0.5)",
            "rgba(82, 84, 163, 0.5)", "rgba(255, 127, 14, 0.5)", "rgba(156, 158, 222, 0.5)",
            "rgba(206, 207, 255, 0.5)"
        ]
    
    ligand_colors = {}
    ligand_color_index = 0
    
    for _, row in df.iterrows():
        ligand = row['Ligand_Symbol']
        receptor = row['Receptor_Symbol']
        tf = row['TF_Symbol']
        tg = row['TG_Symbol']
        att_value = row['Att_Value']

        for node in [ligand, receptor, tf, tg]:
            if node not in node_index:
                node_index[node] = current_index
                nodes.append(node)
                current_index += 1

        if ligand not in ligand_colors:
            ligand_colors[ligand] = path_colors[ligand_color_index % len(path_colors)]
            ligand_color_index += 1

        path_color = ligand_colors[ligand]

        links['source'].extend([node_index[ligand], node_index[receptor], node_index[tf]])
        links['target'].extend([node_index[receptor], node_index[tf], node_index[tg]])
        links['value'].extend([att_value, att_value, att_value])
        links['color'].extend([path_color, path_color, path_color])
    
    node_colors = [node_colorscale[i % len(node_colorscale)] for i in range(len(nodes))]
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="rgba(0,0,0,0)", width=0),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color']
        )
    ))
    
    fig.update_layout(
        title_text=title,
        font_size=font_size,
        autosize=False,
        width=width,
        height=height
    )
    
    return fig


def plot_communication_heatmap(
    df,
    sender_col=0,
    receiver_col=1,
    value_start_col=2,
    colname_suffix='_adj.pickle',
    colors=None,
    figsize=None,  
    xlabel='Features',
    ylabel='Sender->Receiver',
    xtick_rotation=90,
    ytick_rotation=0,
    tight_layout=True,
    font_scale=1.0,
    row_fontsize=8,
    col_fontsize=8,
    title=None,
    cbar=True,
    square=False,
    linewidths=0.1
):
    """
    Draw a heat map of intercellular communication
    """

    sns.set(font_scale=font_scale)
    
    sig_df = df.copy()
    sig_df['combined'] = sig_df.iloc[:, sender_col].astype(str) + '->' + sig_df.iloc[:, receiver_col].astype(str)
    
    new_columns = [col.replace(colname_suffix, '') for col in sig_df.columns[value_start_col:-1]]
    
    heatmap_df = sig_df.iloc[:, value_start_col:-1]
    heatmap_df.columns = new_columns
    heatmap_df.index = sig_df['combined']
    

    if colors is None:
        colors = ['#ffffff','#fde0dd','#fcc5c0','#fa9fb5','#f768a1',
                 '#dd3497','#ae017e','#7a0177','#49006a']
    
    if figsize is None:
        n_rows, n_cols = heatmap_df.shape
        fig_width = min(30, 4 + n_cols * 0.3)  
        fig_height = min(40, 4 + n_rows * 0.2)  
        figsize = (fig_width, fig_height)
    
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heatmap_df,
        cmap=custom_cmap,
        cbar=cbar,
        square=square,
        linewidths=linewidths
    )
    
    ax.set_xlabel(xlabel, fontsize=10*font_scale)
    ax.set_ylabel(ylabel, fontsize=10*font_scale)
    if title:
        ax.set_title(title, fontsize=12*font_scale)
    
    ax.set_xticklabels(ax.get_xticklabels(), 
                      rotation=xtick_rotation, 
                      ha='right' if xtick_rotation != 0 else 'center',
                      fontsize=col_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), 
                      rotation=ytick_rotation,
                      fontsize=row_fontsize)
    
    if tight_layout:
        plt.tight_layout()
    
    return ax


def plot_ct_ct_signaling(coord_df, interaction_df, signaling_item):
    """
    Draw Signal pathway spatial map
    """
    
    cell_types = sorted(coord_df['cell_type'].unique())
    

    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#DAA520', '#8FBC8F', '#4682B4', '#A0522D', '#5F9EA0',
        '#FF69B4', '#708090', '#CD5C5C', '#32CD32', '#800080'
    ]
    
    if len(cell_types) > len(default_colors):
        raise ValueError(f"The number of cell types exceeds the default color count ({len(default_colors)})! Please provide more colors.")
    
    color_map = {ctype: color for ctype, color in zip(cell_types, default_colors)}
    
    coord_df = coord_df.copy()
    coord_df['color'] = coord_df['cell_type'].map(color_map)
    
    centers_df = coord_df.groupby('cell_type')[['x', 'y']].mean().reset_index()
    centers_df.columns = ['cell_type', 'x', 'y']
    
    interaction_strength = {
        (row['Sender'], row['Receiver']): row['Interaction_Strength']
        for _, row in interaction_df.iterrows()
    }
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=coord_df, x='x', y='y', hue='cell_type', palette=color_map,
                    alpha=0.5, s=100, edgecolor='none')
    
    for _, row in centers_df.iterrows():
        plt.scatter(row['x'], row['y'], s=300, marker='o', color=color_map[row['cell_type']])
    
    for (type1, type2), strength in interaction_strength.items():
        if type1 not in color_map or type2 not in color_map:
            continue  
        center1 = centers_df[centers_df['cell_type'] == type1].iloc[0]
        center2 = centers_df[centers_df['cell_type'] == type2].iloc[0]
        
        dx = center2['x'] - center1['x']
        dy = center2['y'] - center1['y']
        
        arrow_width = max(0.5, np.log10(strength + 1) * 10)
        arrow_color = color_map.get(type1, 'gray')
        arrow_alpha = 1.0
        
        arrow = FancyArrowPatch(
            (center1['x'], center1['y']),
            (center2['x'], center2['y']),
            mutation_scale=15,
            color=arrow_color,
            alpha=arrow_alpha,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.3',
            linewidth=arrow_width
        )
        plt.gca().add_patch(arrow)

    handles, labels = plt.gca().get_legend_handles_labels()
    filtered_handles_labels = [
        (handle, label) for handle, label in zip(handles, labels) if "Center" not in label
    ]
    if filtered_handles_labels:
        filtered_handles, filtered_labels = zip(*filtered_handles_labels)
        plt.legend(filtered_handles, filtered_labels, loc='upper left', bbox_to_anchor=(1.05, 1),
                   title="Cell Types")
    else:
        plt.legend([], [], loc='upper left', bbox_to_anchor=(1.05, 1), title="Cell Types")
    

    plt.title('Spatial plot of '+signaling_item)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    
    return plt


def plot_ligand_receptor_expression(exp_matrix, coord_df, ligand_receptor_item):
    """
    Visualize the spatial expression status of ligand receptor pairs
    """
    ligand, receptor = ligand_receptor_item.split("_")
    receptor_parts = receptor.split("-") 
    
    def get_expression_status(cell):
        ligand_exp = exp_matrix.loc[ligand, cell] > 0 if ligand in exp_matrix.index else False
        receptor_exp = all((r in exp_matrix.index and exp_matrix.loc[r, cell] > 0) for r in receptor_parts)
        
        if ligand_exp and receptor_exp:
            return 'Both'
        elif ligand_exp:
            return ligand
        elif receptor_exp:
            return receptor
        else:
            return 'None'
    
    coord_df = coord_df.copy()
    coord_df['expression_status'] = coord_df['cell'].map(get_expression_status)

    expression_color_map = {
        ligand: '#f768a1',     
        receptor: '#74c476',   
        'Both': '#6baed6',     
        'None': '#bdbdbd'     
    }

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=coord_df,
        x='x', y='y',
        hue='expression_status',
        palette=expression_color_map,
        alpha=1, s=100,
        edgecolor='none'
    )

    plt.title(f'Spatial Distribution of {ligand}-{receptor} Expression')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.legend(title='Gene Expression', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    
    return plt


def plot_multiple_gene_expression(exp_matrix, coord_df, gene_list):
    """
    Spatial expression map of multiple genes
    """

    coord_df = coord_df.copy()
    if 'cell' in coord_df.columns:
        coord_df.set_index('cell', inplace=True)

    palette = sns.color_palette("Set2", n_colors=len(gene_list))
    gene_color_map = {gene: color for gene, color in zip(gene_list, palette)}
    gene_color_map['None'] = '#bdbdbd'

    ncols = len(gene_list)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for idx, gene in enumerate(gene_list):
        ax = axes[idx]

        if gene not in exp_matrix.index:
            print(f"[Warning] Gene {gene} not found in expression matrix. Skipping.")
            ax.set_visible(False)
            continue

        coord_df['expression_status'] = 'None'
        expressed_cells = exp_matrix.columns[(exp_matrix.loc[gene] > 0).values]
        coord_df.loc[expressed_cells, 'expression_status'] = gene

        sns.scatterplot(
            data=coord_df,
            x='x', y='y',
            hue='expression_status',
            palette={gene: gene_color_map[gene], 'None': '#bdbdbd'},
            s=100,
            edgecolor='none',
            ax=ax,
            legend=False
        )

        ax.set_title(gene, fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()
    return plt