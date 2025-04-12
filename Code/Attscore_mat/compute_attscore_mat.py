import numpy as np
import pickle as pkl
import ast
import pandas as pd
import random
import json 
import pickle
from tqdm import tqdm
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from tqdm import tqdm
import math

# torch.backends.cudnn.enable =True
# torch.backends.cudnn.benchmark = True
# os.environ['CUDA_LAUNCH_BLOCKING'] ='1'


'''Generate cell node by masking according to cell type '''
def generate_cell_node(sc_type,rc_type,signaling_pathway,gene_cell_mat, sample_labels):
    expressed_gene_insender = list(set(signaling_pathway['Ligand_Symbol']))
    expressed_gene_inreceiver = list(set(signaling_pathway['Receptor_Symbol']))
    gene_insender_positions = [i for i, gene in enumerate(gene_cell_mat.index) if gene in expressed_gene_insender]
    gene_inreceiver_positions = [i for i, gene in enumerate(gene_cell_mat.index) if gene in expressed_gene_inreceiver]
    fixed_idx = list(set(gene_insender_positions + gene_inreceiver_positions))
    masked_gene_cell_mat = pd.DataFrame(0, index=gene_cell_mat.index, columns=gene_cell_mat.columns, dtype=float)
    sample_names_CT1 = sample_labels.loc[sample_labels['Label'] == sc_type, 'Sample Name']
    for gene in expressed_gene_insender:
        if gene in gene_cell_mat.index:
            masked_gene_cell_mat.loc[gene, sample_names_CT1] = gene_cell_mat.loc[gene, sample_names_CT1]

    sample_names_CT2 = sample_labels.loc[sample_labels['Label'] == rc_type, 'Sample Name']
    for gene in expressed_gene_inreceiver:
        if gene in gene_cell_mat.index:
            masked_gene_cell_mat.loc[gene, sample_names_CT2] = gene_cell_mat.loc[gene, sample_names_CT2]
    
    sc_type_now = sc_type.replace(" ", "-")
    rc_type_now = rc_type.replace(" ", "-")
    sc_type_now = sc_type_now.replace("/", "-")     
    rc_type_now = rc_type_now.replace("/", "-")
    
    # node cell fea
    index_temp = []
    for i in range(len(masked_gene_cell_mat.columns)):          
        index_temp.append(i)
    type_temp = []
    for i in range(len(masked_gene_cell_mat.columns)):
        type_temp.append('cell')
    df_index = pd.DataFrame(index_temp, columns = ['id'])
    df_type = pd.DataFrame(type_temp, columns = ['type'])
    df_name = pd.DataFrame(list(masked_gene_cell_mat.columns), columns = ['name'])
    df_cell = pd.concat([df_type, df_index, df_name],  axis = 1)
    
    # find cell node feature 
    cells_all = list(masked_gene_cell_mat.columns)
    cell_fea = []
    fea_temp = [] 
    for i in range(masked_gene_cell_mat.shape[0]):             
        fea_temp.append(0)
    for item in tqdm(cells_all):
        if item in cells_all:
            # df_edge_ligand_receptor['feature'] = list(genes.loc[item, : ])
            cell_fea.append(list(masked_gene_cell_mat.loc[:, item ]))
        else:
            cell_fea.append(fea_temp)
            
    df_cell['feature'] = cell_fea
    
    return masked_gene_cell_mat, df_cell, fixed_idx



def comupte_pathweight(sc_type,rc_type,signaling_pathway,masked_gene_cell_mat,simulate_symbol):
    cells_labels = pd.read_csv('/home/nas2/biod/zhencaiwei/HNNDGI/'+simulate_symbol+'/CT1_CT2_label_str.txt', sep = '\t') 
    sender_list = cells_labels[cells_labels['Label'] == sc_type]
    receiver_list = cells_labels[cells_labels['Label'] == rc_type]
    sender_names = sender_list['Sample Name']
    receiver_names = receiver_list['Sample Name']
    sender_df = masked_gene_cell_mat.loc[:, sender_names] 
    receiver_df = masked_gene_cell_mat.loc[:, receiver_names]
    path_weight = []
    for item in signaling_pathway.values:
        geneA, geneB, geneC, geneD = item[0],item[1], item[2], item[3]
        path_weight.append(sender_df.loc[geneA, :].mean()  + receiver_df.loc[geneB, :].mean() + 
                        receiver_df.loc[geneC, :].mean() + receiver_df.loc[geneD, :].mean() )

    # row normalize
    total_weight = sum(path_weight)
    path_weight = list(path_weight / total_weight)
    float_path_weight = [float('%.4f' % i) for i in path_weight]
    return float_path_weight




def spot_comupte_pathweight(sc_type,rc_type,signaling_pathway,masked_gene_cell_mat,cells_labels):
    if (sc_type == 'CT1' and rc_type == 'CT3') or (sc_type == 'CT3' and rc_type == 'CT1'):
        float_path_weight = [1]*len(signaling_pathway)
    else:
        cells_labels = cells_labels 
        sender_list = cells_labels[cells_labels['Label'] == sc_type]
        receiver_list = cells_labels[cells_labels['Label'] == rc_type]
        sender_names = sender_list['Sample Name']
        receiver_names = receiver_list['Sample Name']
        sender_df = masked_gene_cell_mat.loc[:, sender_names] 
        receiver_df = masked_gene_cell_mat.loc[:, receiver_names]
        path_weight = []
        for item in signaling_pathway.values:
            geneA, geneB, geneC, geneD = item[0],item[1], item[2], item[3]
            path_weight.append(sender_df.loc[geneA, :].mean()*3  + receiver_df.loc[geneB, :].mean() + 
                            receiver_df.loc[geneC, :].mean() + receiver_df.loc[geneD, :].mean() )

        
        # 75 percentile normalize + row normalize
        percentile_75 = np.percentile(path_weight, 75)
        new_data = [x * 2 if x > percentile_75 else x / 2 for x in path_weight]
        path_weight = list(new_data / np.sum(new_data))
        float_path_weight = [float('%.4f' % i) for i in path_weight]
        
    return float_path_weight



def generate_adj_matrix(gene_cell_mat, signaling_pathway, path_nt, coordinates_df):
    """
    Process the signaling pathway data, generate the ligand-receptor-tf-tg adjacency matrices, and save the results.
    
    Parameters:
    - gene_cell_mat: DataFrame containing gene expression data for cells.
    - signaling_pathway: DataFrame containing signaling pathway information.
    - path_nt: string, the outputs dir.
    - coordinates_df: DataFrame containing coordinates of cells.

    Saves:
    - adjacency matrices (pickle files) for each ligand-receptor-TF-TG combination.
    """

    ligands_cells_num = {}
    receptors_cells_num = {}

    for ligand in tqdm(signaling_pathway['Ligand_Symbol']):
        row = gene_cell_mat.loc[ligand]
        non_zero_columns = list(row[row != 0].index)
        positions = [row.index.get_loc(col) for col in non_zero_columns]
        ligands_cells_num[ligand] = positions

    for i in tqdm(range(len(signaling_pathway))):
        TG = signaling_pathway.loc[i, 'TG_Symbol']
        receptor_symbol = signaling_pathway.loc[i, 'Receptor_Symbol']
        TF_symbol = signaling_pathway.loc[i, 'TF_Symbol']
        key = receptor_symbol + TF_symbol + TG

        row = gene_cell_mat.loc[TG]
        non_zero_columns = list(row[(row != 0) & (gene_cell_mat.loc[receptor_symbol] != 0) & (gene_cell_mat.loc[TF_symbol] != 0)].index)
        positions = [row.index.get_loc(col) for col in non_zero_columns]
        receptors_cells_num[key] =  positions

    
    Lcell = ligands_cells_num
    Rcell = receptors_cells_num

    num_cells = gene_cell_mat.shape[1]
    print(f"There are {num_cells} cells")

    for ligand, receptor, TF, TG in signaling_pathway.itertuples(index=False):
        key = receptor + TF + TG
        l_coordinates = Lcell.get(ligand, [])
        r_coordinates = Rcell.get(key, [])

        adj_matrix = lil_matrix((num_cells, num_cells), dtype=np.float32)

        for l_cell in l_coordinates:
            for r_cell in r_coordinates:
                l_index = l_cell
                r_index = r_cell
                expression = (
                    gene_cell_mat.loc[ligand, 'spot_' + str(l_index)] +
                    gene_cell_mat.loc[receptor, 'spot_' + str(r_index)] +
                    gene_cell_mat.loc[TF, 'spot_' + str(r_index)] +
                    gene_cell_mat.loc[TG, 'spot_' + str(r_index)]
                )

                x_coordinates = coordinates_df.loc[l_index, ['x', 'y']]
                y_coordinates = coordinates_df.loc[r_index, ['x', 'y']]
                distance = np.linalg.norm(x_coordinates - y_coordinates)

                if distance <= 5:
                    expression_up = math.ceil(expression)
                else:
                    expression_up = 0

                adj_matrix[l_index, r_index] = expression_up
                adj_matrix[r_index, l_index] = expression_up  

        adj_matrix = adj_matrix.tocoo()
        pickle_filename = path_nt + f"clrfgc_adj/{ligand}_{receptor}_{TF}_{TG}_adj.pickle"
        with open(pickle_filename, 'wb') as file:
            pickle.dump(adj_matrix, file)

        print(f"{pickle_filename} has been saved") 